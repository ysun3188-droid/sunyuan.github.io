# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import threading
import os
import csv

import time
from common.SingleStockExecution import SingleStockExecution
from common.OrderBook import OrderBook
from common.SingleStockOrder import SingleStockOrder
class ExchangeSimulator:
    
    def __init__(self, marketData_2_exchSim_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call ExchSim.init" % (os.getpid(),))
        self.order_book = OrderBook()
        self.exec_log_file = "executions_log.csv"
        # 初始化文件头
        with open(self.exec_log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["date", "ticker", "timestamp", "execID", "orderID",
                             "direction", "price", "size", "commission"])

        t_md = threading.Thread(name='exchsim.on_md', target=self.consume_md,
                                args=(marketData_2_exchSim_q, exchSim_2_platform_execution_q))
        t_md.start()
        
        t_order = threading.Thread(name='exchsim.on_order', target=self.consume_order,
                                   args=(marketData_2_exchSim_q, platform_2_exchSim_order_q, ))
        t_order.start()

    def log_execution_to_file(self, execution):
        with open(self.exec_log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(execution.outputAsArray())

    def consume_md(self, marketData_2_exchSim_q, exchSim_2_platform_execution_q):
        while True:
            res = marketData_2_exchSim_q.get()
            print(f'[{os.getpid()}]ExchSim.consume_md {res.outputAsArray()}')

            executions = self.order_book.match_orders(res)
            for exec in executions:
                exchSim_2_platform_execution_q.put(exec)
                print(f'[{os.getpid()}]ExchSim.executed_trade {exec.outputAsArray()}')
                self.log_execution_to_file(exec)

    def consume_order(self, marketData_2_exchSim_q, platform_2_exchSim_order_q):
        while True:
            snapshot = marketData_2_exchSim_q.get()
            msg = platform_2_exchSim_order_q.get()
            if isinstance(msg, tuple) and msg[0] == "CANCEL":
                self.order_book.cancel_order(msg[1])
            elif isinstance(msg, SingleStockOrder):
                self.order_book.add_order(msg, snapshot)
                print(f'[{os.getpid()}]ExchSim.on_order {msg.outputAsArray()}')

    # def produce_execution(self, order, exchSim_2_platform_execution_q):
    #     execution = SingleStockExecution(order.ticker, order.date, time.asctime(time.localtime(time.time())))
    #     exchSim_2_platform_execution_q.put(execution)
    #     print(f'[{os.getpid()}]ExchSim.produce_execution {execution.outputAsArray()}')

