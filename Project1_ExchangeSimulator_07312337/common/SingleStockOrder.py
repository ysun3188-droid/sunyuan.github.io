# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:47:43 2019

@author: hongsong chou
"""

class SingleStockOrder():
    
    def __init__(self, ticker, date, submissionTime):
        self.orderID = 0
        self.ticker = ticker
        self.date = date
        self.submissionTime = submissionTime
        self.currStatusTime = None
        self.currStatus = None #"New", "Filled", "PartiallyFilled", "Cancelled"
        self.direction = None
        self.price = None
        self.size = None
        self.type = None #"MLO", "LO", "MO", "TWAP"

    def outputAsArray(self):
        output = []
        output.append(self.date)
        output.append(self.ticker)
        output.append(self.submissionTime)
        output.append(self.orderID)
        output.append(self.currStatus)
        output.append(self.currStatusTime)
        output.append(self.direction)
        output.append(self.price)
        output.append(self.size)
        output.append(self.type)
        
        return output
    
    def copyOrder(self):
        returnOrder = SingleStockOrder(self.ticker, self.date)
        returnOrder.orderID = self.orderID
        returnOrder.submissionTime = self.submissionTime
        returnOrder.currStatusTime = self.currStatusTime
        returnOrder.currStatus = self.currStatus
        returnOrder.direction = self.direction
        returnOrder.price = self.price
        returnOrder.size = self.size
        returnOrder.type = self.type
        
        return returnOrder

