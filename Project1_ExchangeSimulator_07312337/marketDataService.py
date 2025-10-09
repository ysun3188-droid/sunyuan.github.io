# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:12:21 2020

@author: hongsong chou
"""

import time
import random
import os
import pandas as pd
from common.OrderBookSnapshot_FiveLevels import OrderBookSnapshot_FiveLevels

class MarketDataService:

    def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q, csv_file_path="data/2603_md_202108_202108.csv"):
    # def __init__(self, marketData_2_exchSim_q, marketData_2_platform_q, csv_file_path="test_data.csv"):
        print("[%d]<<<<< call MarketDataService.init" % (os.getpid(),))
        self.marketData_2_exchSim_q = marketData_2_exchSim_q
        self.marketData_2_platform_q = marketData_2_platform_q
        self.csv_file_path = csv_file_path
        self.produce_market_data()

    def produce_market_data(self):
        df = pd.read_csv(self.csv_file_path)
        df = df.iloc[:, 1:]  # Drop first unnamed column
        df = df[df['date'] == '2021-08-02']
        for idx, row in df.iterrows():
            # 生成五档行情
            bid_prices = [row[f'BP{i}'] for i in range(1, 6)]
            ask_prices = [row[f'SP{i}'] for i in range(1, 6)]
            bid_sizes = [row[f'BV{i}'] for i in range(1, 6)]
            ask_sizes = [row[f'SV{i}'] for i in range(1, 6)]
            lastPx = row['lastPx'] if not pd.isna(row['lastPx']) else None
            size = row['size'] if not pd.isna(row['size']) else None
            if ask_prices[0] == 0:
                continue
            ob_snapshot = OrderBookSnapshot_FiveLevels(
                ticker="2603",
                date=row['date'],
                timeStamp=str(row['time']),
                bidPrice=bid_prices,
                askPrice=ask_prices,
                bidSize=bid_sizes,
                askSize=ask_sizes,
                lastPx=lastPx,
                size=size
            )

            if ob_snapshot.initializationFlag:
                print('[%d]MarketDataService>>>produce_quote' % (os.getpid()))
                print(ob_snapshot.outputAsArray())
                self.marketData_2_exchSim_q.put(ob_snapshot)
                self.marketData_2_platform_q.put(ob_snapshot)

            time.sleep(0.01)



if __name__ == '__main__':
    from multiprocessing import Queue

    marketData_2_exchSim_q = Queue()
    marketData_2_platform_q = Queue()

    mds = MarketDataService(marketData_2_exchSim_q, marketData_2_platform_q, csv_file_path="2603_md_202108_202108.csv")
