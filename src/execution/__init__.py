import logging
from typing import Any, NamedTuple

import ccxt.async_support as ccxt

from src.core.risk import RiskPlan

log = logging.getLogger(__name__)


class Execution(NamedTuple):
    price: float  # average fill
    quantity: float  # coins held after fees (what the OCO protects and an exit sells)
    cost_usd: float
    stop_loss: float  # re-anchored to the fill, same distances as the plan
    take_profit: float
    ref: str  # exchange OCO orderListId


class BinanceDemoBroker:
    """Real orders on Binance Demo Trading: real prices, fake funds.

    Entry = market buy + exchange-side OCO (LIMIT_MAKER take-profit / STOP_LOSS market stop),
    so stops fire even while the bot is offline. The exchange is the source of truth for exits.
    """

    def __init__(self, api_key: str, secret: str) -> None:
        self.ex: Any = ccxt.binance({"apiKey": api_key, "secret": secret, "enableRateLimit": True})
        self.ex.enable_demo_trading(True)

    async def free_usdt(self) -> float:
        balance = await self.ex.fetch_balance()
        return float(balance["free"].get("USDT", 0.0))

    async def enter(self, symbol: str, plan: RiskPlan, signal_price: float) -> Execution:
        await self.ex.load_markets()
        min_cost = self.ex.market(symbol)["limits"]["cost"]["min"] or 0.0
        if plan.position_size_usd < 2 * min_cost:  # both OCO legs must clear min notional too
            raise ValueError(f"{symbol} size ${plan.position_size_usd:.2f} below exchange minimum")
        order = await self.ex.create_market_buy_order_with_cost(symbol, plan.position_size_usd)
        base = self.ex.market(symbol)["base"]
        fee_in_base = sum(f["cost"] for f in order.get("fees") or [] if f.get("currency") == base)
        # Truncate to the lot step (rounding up would try to sell coins we don't have).
        qty = float(self.ex.amount_to_precision(symbol, order["filled"] - fee_in_base))
        price = float(order["average"])
        sl = price - (signal_price - plan.stop_loss)
        tp = price + (plan.take_profit - signal_price)
        try:
            ref = await self.place_oco(symbol, qty, sl, tp)
        except Exception:
            # Never hold coins without a stop: undo the entry, then surface the error.
            log.exception("OCO failed for %s; selling the fresh position back", symbol)
            await self.ex.create_market_sell_order(symbol, qty)
            raise
        return Execution(price, qty, float(order["cost"]), sl, tp, ref)

    async def place_oco(self, symbol: str, qty: float, sl: float, tp: float) -> str:
        resp = await self.ex.privatePostOrderListOco(
            {
                "symbol": self.ex.market_id(symbol),
                "side": "SELL",
                "quantity": self.ex.amount_to_precision(symbol, qty),
                "aboveType": "LIMIT_MAKER",
                "abovePrice": self.ex.price_to_precision(symbol, tp),
                "belowType": "STOP_LOSS",  # market on trigger: guaranteed exit, not a price
                "belowStopPrice": self.ex.price_to_precision(symbol, sl),
            }
        )
        return str(resp["orderListId"])

    async def check(self, symbol: str, ref: str) -> tuple[str, float] | None:
        """(reason, fill price) once the OCO finished; ("cancelled", 0) if it ended unfilled."""
        lst = await self.ex.privateGetOrderList({"orderListId": ref})
        if lst["listOrderStatus"] != "ALL_DONE":
            return None
        for leg in lst["orders"]:
            order = await self.ex.fetch_order(str(leg["orderId"]), symbol)
            if order["filled"]:
                reason = "stop_loss" if order["info"]["type"] == "STOP_LOSS" else "take_profit"
                return reason, float(order["average"])
        return "cancelled", 0.0

    async def exit(self, symbol: str, qty: float, ref: str) -> float:
        """Strategy exit: cancel the OCO, market-sell the position. Returns the fill price."""
        try:
            await self.ex.privateDeleteOrderList(
                {"symbol": self.ex.market_id(symbol), "orderListId": ref}
            )
        except ccxt.OrderNotFound:
            pass  # OCO already done; the sell below fails loudly if the coins are gone too
        order = await self.ex.create_market_sell_order(symbol, qty)
        return float(order["average"])

    async def close(self) -> None:
        await self.ex.close()
