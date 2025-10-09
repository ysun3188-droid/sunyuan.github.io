import time
from dataclasses import dataclass
from typing import List

from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution


@dataclass
class MyOrder:
    order: SingleStockOrder
    queue_ahead: int
    remaining: int
    idx: int


class OrderBook:

    def __init__(self, log_path: str = "trade_log.txt"):
        self._my_orders: List[MyOrder] = []
        self._last_snapshot = None
        self._pending_ticks: List = []
        self._log_path = log_path
        self.order_index = 0
        open(self._log_path, "w").close()


    def add_order(self, order: SingleStockOrder, snapshot):
        queue = self._queue_before_me(order, snapshot)
        self._my_orders.append(MyOrder(order=order, queue_ahead=queue, remaining=order.size, idx=self.order_index))
        self.order_index += 1

    def cancel_order(self, oid):
        for m in self._my_orders:
            if m.order.orderID == oid and m.remaining > 0:
                m.remaining = 0
                break
    @staticmethod
    def _queue_before_me(order, snap):
        if order.direction == "BUY":
            for i in range(1, 6):
                if getattr(snap, f"bidPrice{i}") == order.price:
                    return getattr(snap, f"bidSize{i}")
        else:
            for i in range(1, 6):
                if getattr(snap, f"askPrice{i}") == order.price:
                    return getattr(snap, f"askSize{i}")
        return 0

    def match_orders(self, md_row):
        fills = []
        if self._is_empty_line(md_row):
            self._pending_ticks.append(md_row)
            return fills

        if self._pending_ticks:
            self._pending_ticks.append(md_row)
            print('大单', sep='')
            fills.extend(self._handle_active_tick_group(self._pending_ticks))
            self._pending_ticks.clear()
        else:
            if self._last_snapshot is not None and md_row.lastPx:
                fills.extend(self._handle_active_tick_group([md_row]))

        fills.extend(self._handle_passive_cross(md_row))
        self._last_snapshot = md_row
        return fills

    @staticmethod
    def _is_empty_line(md):
        return md.askPrice1 == 0 and md.bidPrice1 == 0

    def _handle_active_tick_group(self, ticks):
        fills = []
        if self._last_snapshot is None:
            return fills

        side, total_qty, px_bucket = self._aggregate_trade_group(ticks)
        if side is None or total_qty == 0:
            return fills

        sweep_prices = sorted(px_bucket.keys()) if side == "BUY" else sorted(px_bucket.keys(), reverse=True)
        # idx_pre  = -1
        for mo in list(self._my_orders):
            # assert mo.idx > idx_pre, "idx 顺序有误"
            # idx_pre = mo.idx
            if total_qty == 0:
                break

            hit = (
                side == "BUY"  and mo.order.direction == "SELL" and mo.order.price <= max(sweep_prices)
            ) or (
                side == "SELL" and mo.order.direction == "BUY"  and mo.order.price >= min(sweep_prices)
            )
            if not hit:
                continue

            better_price_count = 0
            for p, p_count in px_bucket.items():
                if (mo.order.direction == "SELL" and p < mo.order.price) or (mo.order.direction == "BUY" and p > mo.order.price):
                    better_price_count += p_count
            total_qty -= better_price_count

            consumed = min(mo.queue_ahead, total_qty)
            mo.queue_ahead -= consumed
            total_qty      -= consumed
            if total_qty == 0:
                break

            fill_qty = min(mo.remaining, total_qty)
            fills.append(self._build_exec(mo.order, fill_qty, mo.order.price))

            data_rows = [f"{t.timeStamp}, lastPx={t.lastPx}, size={int(t.size)}"
                         for t in ticks if t.lastPx]
            self._log_trade(
                idx = mo.idx,
                trade_type="Market order",
                data_rows=data_rows,
                my_order=mo.order,
                remaining = mo.remaining,
                queue_before=mo.queue_ahead + consumed,
                fill_qty=fill_qty,
                fill_price=mo.order.price
            )

            mo.remaining -= fill_qty
            total_qty    -= fill_qty
            if mo.remaining == 0:
                self._my_orders.remove(mo)
        if fills:
            print('Market order', end='')
        return fills

    def _aggregate_trade_group(self, ticks):
        total = 0
        bucket = {}
        for t in ticks:
            if t.size and t.lastPx:
                qty = int(t.size)
                px = t.lastPx
                total += qty
                bucket[px] = bucket.get(px, 0) + qty
        if total == 0 or self._last_snapshot is None:
            return None, 0, bucket
        first_px = ticks[0].lastPx
        if first_px >= self._last_snapshot.askPrice1:
            side = "BUY"
        elif first_px <= self._last_snapshot.bidPrice1:
            side = "SELL"
        else:
            side = None
        return side, total, bucket

    def _handle_passive_cross(self, snap):
        fills = []
        for mo in list(self._my_orders):
            if mo.remaining == 0:
                self._my_orders.remove(mo)
                continue

            if mo.order.direction == "BUY":
                fills.extend(self._eat_book(
                    mo, snap, side="BUY",
                    book_prices=[getattr(snap, f"askPrice{i}") for i in range(1, 6)],
                    book_sizes=[getattr(snap, f"askSize{i}")  for i in range(1, 6)],
                    cross_cond=lambda p: p <= mo.order.price
                ))
            else:
                fills.extend(self._eat_book(
                    mo, snap, side="SELL",
                    book_prices=[getattr(snap, f"bidPrice{i}") for i in range(1, 6)],
                    book_sizes=[getattr(snap, f"bidSize{i}")  for i in range(1, 6)],
                    cross_cond=lambda p: p >= mo.order.price
                ))
            if mo.remaining == 0:
                self._my_orders.remove(mo)
        if fills:
            print('Limit order', end='')
        return fills

    def _eat_book(self, mo, snap, side, book_prices, book_sizes, cross_cond):
        fills = []
        for p, s in zip(book_prices, book_sizes):
            if s == 0 or not cross_cond(p):
                break
            take_qty = min(mo.remaining, s)
            fills.append(self._build_exec(mo.order, take_qty, p))
            if take_qty >0:
                mo.queue_ahead = 0
            snap_line = f"snapshot time={snap.timeStamp}, SP1={snap.askPrice1}, SV1={snap.askSize1}, BP1={snap.bidPrice1}, BV1={snap.bidSize1}"
            self._log_trade(
                idx=mo.idx,
                trade_type="Limit order",
                data_rows=[snap_line],
                my_order=mo.order,
                remaining=mo.remaining,
                queue_before=mo.queue_ahead,
                fill_qty=take_qty,
                fill_price=p
            )
            mo.remaining -= take_qty
            if mo.remaining == 0:
                break
        return fills

    @staticmethod
    def _build_exec(order, qty, price):
        ex = SingleStockExecution(order.ticker, order.date, time.asctime())
        ex.execID = order.orderID
        ex.orderID = order.orderID
        ex.direction = order.direction
        ex.price = round(price, 2)
        ex.size = qty
        return ex

    def _log_trade(self, *, idx, trade_type, data_rows, my_order, remaining, queue_before, fill_qty, fill_price):
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("─" * 52 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]  {trade_type}\n")
            f.write("Data rows:\n")
            for line in data_rows:
                f.write(f"  {line}\n")
            f.write(f"My order {idx}:  {my_order.direction} {my_order.price} × {my_order.size}   | queue_ahead={queue_before} |   remaining={remaining} \n")
            f.write(f"Trade result:  {my_order.direction} {fill_price} × {fill_qty}\n")


            
if __name__ == '__main__':
    import pandas as pd
    from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels

    CSV_PATH = "../data/test_data.csv"
    df = pd.read_csv(CSV_PATH)

    ob = OrderBook()
    first_complete_snapshot = None

    for i, row in df.iterrows():
        snap = OrderBookSnapshot_FiveLevels(
            ticker="2603",
            date=row['date'],
            timeStamp=str(row['time']),
            bidPrice=[row[f'BP{i}'] for i in range(1, 6)],
            askPrice=[row[f'SP{i}'] for i in range(1, 6)],
            bidSize=[row[f'BV{i}'] for i in range(1, 6)],
            askSize=[row[f'SV{i}'] for i in range(1, 6)],
            lastPx=row['lastPx'] if not pd.isna(row['lastPx']) else None,
            size=row['size'] if not pd.isna(row['size'])   else None
        )

        if first_complete_snapshot is None and not ob._is_empty_line(snap):
            # if snap.timeStamp != "90002622": continue
            first_complete_snapshot = snap
            from common.SingleStockOrder import SingleStockOrder
            my_order = SingleStockOrder("2603", row['date'], time.asctime())
            my_order.orderID = 1
            my_order.direction = "SELL"
            my_order.price = 13250
            my_order.size = 100
            ob.add_order(my_order, snapshot=first_complete_snapshot)
            print(f">>> 已挂单 SELL {my_order.price} × {my_order.size}\n")

        exes = ob.match_orders(snap)
        for e in exes:
            print(f"[{snap.timeStamp}] 成交 →", e.outputAsArray())