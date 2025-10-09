# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:39:28 2019

@author: hongs
"""

import pandas as pd
from common.OrderBookSnapshot import OrderBookSnapshot

class OrderBookSnapshot_FiveLevels(OrderBookSnapshot):
    bidPrice1, bidPrice2, bidPrice3, bidPrice4, bidPrice5 = None, None, None, None, None
    askPrice1, askPrice2, askPrice3, askPrice4, askPrice5 = None, None, None, None, None
    bidSize1, bidSize2, bidSize3, bidSize4, bidSize5 = None, None, None, None, None
    askSize1, askSize2, askSize3, askSize4, askSize5 = None, None, None, None, None
    
    initializationFlag = False
    
    outputCols = ['ticker','date','time', \
                  'askPrice5','askPrice4','askPrice3','askPrice2','askPrice1', \
                  'bidPrice1','bidPrice2','bidPrice3','bidPrice4','bidPrice5', \
                  'askSize5','askSize4','askSize3','askSize2','askSize1', \
                  'bidSize1','bidSize2','bidSize3','bidSize4','bidSize5']

    def __init__(self, ticker, date, timeStamp, bidPrice, askPrice, bidSize, askSize, lastPx, size):
        super().__init__(ticker, date, timeStamp)
                
        if (bidPrice is None) or (askPrice is None) or (bidSize is None) or (askSize is None):
            print("In OrderBookSnapshot_FiveLevels: bidPrice, askPruce, bidSize, askSize empty.")
            return
        elif (len(bidPrice) != 5) or (len(askPrice) != 5) or (len(bidSize) != 5) or (len(askSize) != 5):
            print("In OrderBookSnapshot_FiveLevels: bidPrice, askPruce, bidSize, askSize sizes not match.")
            return
        else:
            self.askPrice1, self.askPrice2, self.askPrice3, self.askPrice4, self.askPrice5 = \
                askPrice[0], askPrice[1], askPrice[2], askPrice[3], askPrice[4],
            self.bidPrice1, self.bidPrice2, self.bidPrice3, self.bidPrice4, self.bidPrice5 = \
                bidPrice[0], bidPrice[1], bidPrice[2], bidPrice[3], bidPrice[4],
            self.askSize1, self.askSize2, self.askSize3, self.askSize4, self.askSize5 = \
                askSize[0], askSize[1], askSize[2], askSize[3], askSize[4],
            self.bidSize1, self.bidSize2, self.bidSize3, self.bidSize4, self.bidSize5 = \
                bidSize[0], bidSize[1], bidSize[2], bidSize[3], bidSize[4],
            self.initializationFlag = True,
            self.lastPx = lastPx
            self.size = size
            return

    def outputAsDataFrame(self):
        if self.initializationFlag == False:
            return None
        else:
            outputLine = []
            outputLine.append(self.ticker)
            outputLine.append(self.date)
            outputLine.append(self.timeStamp)
            outputLine.append(self.askPrice5)
            outputLine.append(self.askPrice4)
            outputLine.append(self.askPrice3)
            outputLine.append(self.askPrice2)
            outputLine.append(self.askPrice1)
            outputLine.append(self.bidPrice1)
            outputLine.append(self.bidPrice2)
            outputLine.append(self.bidPrice3)
            outputLine.append(self.bidPrice4)
            outputLine.append(self.bidPrice5)
            outputLine.append(self.askSize5)
            outputLine.append(self.askSize4)
            outputLine.append(self.askSize3)
            outputLine.append(self.askSize2)
            outputLine.append(self.askSize1)
            outputLine.append(self.bidSize1)
            outputLine.append(self.bidSize2)
            outputLine.append(self.bidSize3)
            outputLine.append(self.bidSize4)
            outputLine.append(self.bidSize5)
            oneLine = pd.DataFrame(data = [outputLine], columns = self.outputCols)
            
            return oneLine
            
    def outputAsArray(self):
        if not self.initializationFlag:
            return None
        else:
            return [
                self.ticker,
                self.date,
                self.timeStamp,
                f"sp1: {self.askPrice1}",
                f"sv1: {self.askSize1}",
                f"bp1: {self.bidPrice1}",
                f"bv1: {self.bidSize1}"
            ]


