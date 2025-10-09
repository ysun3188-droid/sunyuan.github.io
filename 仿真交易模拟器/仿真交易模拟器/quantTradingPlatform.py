# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:15:48 2020

@author: hongsong chou sun yuan
"""

import threading
import os
from QuantStrategy import QuantStrategy

class TradingPlatform:
    quantStrat = None
    
    def __init__(self, marketData_2_platform_q, platform_2_exchSim_order_q, exchSim_2_platform_execution_q):
        print("[%d]<<<<< call Platform.init" % (os.getpid(),))
        
        #Instantiate individual strategies
        self.quantStrat = QuantStrategy("tf_1","quantStrategy","hongsongchou","JBF_3443","20210802")

        t_md = threading.Thread(name='platform.on_marketData', target=self.consume_marketData, args=(platform_2_exchSim_order_q, marketData_2_platform_q,))
        t_md.start()
        #
        # t_exec = threading.Thread(name='platform.on_exec', target=self.handle_execution, args=(exchSim_2_platform_execution_q, ))
        # t_exec.start()

    def consume_marketData(self, platform_2_exchSim_order_q, marketData_2_platform_q):
        print('[%d]Platform.consume_marketData' % (os.getpid(),))
        while True:
            snapshot = marketData_2_platform_q.get()
            print('[%d] Platform.on_md' % (os.getpid()))
            print(snapshot.outputAsArray())
            self.quantStrat.on_market_data(
                snapshot,
                send_order=lambda o: platform_2_exchSim_order_q.put(o),
                send_cancel=lambda o: platform_2_exchSim_order_q.put(("CANCEL", o.orderID))
            )



