import tushare as ts
import pandas as pd
import pickle


class stock_data:
    def __init__(self, code, start="2010-01-01", ktype="W", threshold=1.0):
        self.code = code
        self.start = start
        self.ktype = ktype
        self.threshold = threshold
        self.__statistic()

    def __statistic(self):
        self.detail_df = ts.get_k_data(self.code,
                                       start=self.start, ktype=self.ktype)
        wc = self.detail_df.close.copy()
        wc2 = wc[1:].copy()
        wc2.reset_index(drop=True, inplace=True)
        self.wcp = (wc2 / wc)
        self.__find_increase(self.wcp > self.threshold)

    def __find_increase(self, wcp_arr):
        self.length_pair = []
        self.max_length = 0
        length = 0
        start = 0
        pre = False
        for i, v in enumerate(wcp_arr):
            if v:
                length += 1
                if not pre:
                    start = i
            else:
                if pre and length > 2:
                    self.max_length = max(self.max_length, length)
                    self.length_pair.append((start, i))
                length = 0
            pre = v

if __name__ == '__main__':
    """data_arr = []
    threshold = 1.0
    ac_df = ts.get_area_classified()
    for stock in ac_df.code:
        if str.startswith(stock, "300"):
            continue
        data = stock_data(stock)
        data_arr.append(data)
    with open("week_data.pkl", "wb") as f:
        pickle.dump(data_arr, f)
    """
    with open("week_data.pkl", "rb") as f:
        data_arr = pickle.load(f)
    #找到max_length大于5的
    for stock in data_arr:
        if stock.max_length > 5:
            #寻找长期上升的，在length_pair里，统计上升的星期数
            total = 0
            for start, end in stock.length_pair:
                total += end-start
            if total > 80:
                print(stock.code, total)
                stock.detail_df.plot()