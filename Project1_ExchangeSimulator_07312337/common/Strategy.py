#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on Thu Jun 19 18:54:05 2019

@author: hongsong chou
"""

class Strategy():
    
    def __init__(self, stratID, stratName, stratAuthor):
        self.__stratID = stratID #private field
        self.__stratName = stratName #private field
        self.__stratAuthor = stratAuthor #private field
        
    def getStratID(self):
        return self.__stratID
    
    def getStratName(self):
        return self.__stratName
    
    def getStratAuthor(self):
        return self.__stratAuthor
