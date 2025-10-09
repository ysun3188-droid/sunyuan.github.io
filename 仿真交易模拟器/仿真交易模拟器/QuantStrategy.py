#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 20 10:26:05 2020

@author: hongsong chou sun yuan
"""
import random
import os
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels
from common.Strategy import Strategy
from common.SingleStockOrder import SingleStockOrder
from common.SingleStockExecution import SingleStockExecution
from collections import deque
import time
class QuantStrategy(Strategy):
    
    def __init__(self, stratID, stratName, stratAuthor, ticker, day, interval_sec=50, lookback=50):
        super(QuantStrategy, self).__init__(stratID, stratName, stratAuthor) #call constructor of parent
        self.ticker = ticker #public field
        self.day = day #public field
        self.orderID_counter = 1
        self.interval = interval_sec
        self.lookback = lookback
        self.md_buffer = deque(maxlen=lookback)
        self.active_orders = {}
        self._last_submit = 0
    def getStratDay(self):
        return self.day

    def on_market_data(self, snapshot, send_order, send_cancel):

        # 1) 记录行情
        self.md_buffer.append(snapshot)
        print()
        # 2) 定时挂单
        now = int(snapshot.timeStamp)
        if not hasattr(self, "_last_submit") or (now - self._last_submit)/1000 >= self.interval:
            self._last_submit = now
            for side, price in (("BUY", snapshot.bidPrice1),
                                ("SELL", snapshot.askPrice1)):
                order = SingleStockOrder(self.ticker, snapshot.date, time.asctime())
                order.orderID = int(now * 1000)
                order.direction = side
                order.price = price
                order.size = 100 if random.random() < 0.2 else 200

                order.type = "LO"
                send_order(order)
                self.active_orders[order.orderID] = {
                    'order': order,
                    'submit_ts': now,
                    'queue': snapshot.bidSize1 if side == "SELL" else snapshot.askSize1
                }
        if len(self.md_buffer) >= 2:
            turnover_lambda = self._estimate_lambda(snapshot)
            for oid, info in list(self.active_orders.items()):
                qtt = info['queue'] / (turnover_lambda or 1)
                if now - info['submit_ts'] > qtt:
                    # cancel order
                    send_cancel(info['order'])
                    del self.active_orders[oid]
    def on_execution(self, exec_report):
        self.active_orders.pop(exec_report.orderID, None)

    def _estimate_lambda(self, latest_snap):
        """ bid1/ask1 transaction speed"""
        vols, t0, t1 = 0, None, None
        for md in reversed(self.md_buffer):
            if md.lastPx is None:
                continue
            if md.lastPx in (latest_snap.bidPrice1, latest_snap.askPrice1):
                vols += md.size
                if t0 is None:
                    t0 = md.timeStamp
            if t1 is None:
                t1 = md.timeStamp
        if not vols or not t0 or not t1 or t0 == t1:
            return 0
        span = (int(t1) - int(t0)) / 100
        return vols / max(span, 1)
