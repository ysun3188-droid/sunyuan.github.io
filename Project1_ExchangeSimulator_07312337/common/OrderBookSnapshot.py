# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:39:28 2019

@author: hongs
"""

class OrderBookSnapshot():
    ticker = None
    date = None
    timeStamp = None
    
    def __init__(self, ticker, date, timeStamp):
        self.ticker = ticker
        self.date = date
        self.timeStamp = timeStamp
