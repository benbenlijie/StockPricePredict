import tushare as ts
import numpy
import pandas
import math

from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Embedding

def prepare_stock_data(stock):
    stock_data = ts.get_k_data(stock)
    stock_data = stock_data.sort_values("date")
    stock_data = stock_data.set_index(["date"])
    stock_data = calculate_macd(stock_data)
    stock_data = calculate_change_aver(stock_data)
    return stock_data

def calculate_macd(df):
    new_data = df.sort_index()
    start_index = 0
    new_data.ix[start_index, 'ema_12'] = new_data.ix[start_index, 'close']
    new_data.ix[start_index, 'ema_26'] = new_data.ix[start_index, 'close']
    new_data.ix[start_index, 'ema_9'] = 0
    for i in range(1, len(new_data.index)):
    #print(i)
        new_data.ix[i, 'ema_12'] = new_data.ix[i-1, 'ema_12'] * 11 / 13 + new_data.ix[i, "close"] * 2 / 13
        new_data.ix[i, 'ema_26'] = new_data.ix[i-1, 'ema_26'] * 25 / 27 + new_data.ix[i, 'close'] * 2 / 27
        new_data.ix[i, 'diff'] = new_data.ix[i, 'ema_12'] - new_data.ix[i, 'ema_26']
        new_data.ix[i, 'ema_9'] = new_data.ix[i-1, 'ema_9'] * 8 / 10 + new_data.ix[i, "diff"] * 2 / 10
        new_data.ix[i, 'bar'] = 2 * (new_data.ix[i, 'diff'] - new_data.ix[i, 'ema_9'])
    return new_data

def calculate_change_aver(df):
    new_data = df.sort_index()
    new_data.ix[0, 'ma5'] = new_data.ix[0, "close"]
    new_data.ix[0, 'ma10'] = new_data.ix[0, "close"]
    new_data.ix[0, 'ma20'] = new_data.ix[0, "close"]
    new_data.ix[0, 'v_ma5'] = new_data.ix[0, "volume"]
    new_data.ix[0, 'v_ma10'] = new_data.ix[0, "volume"]
    new_data.ix[0, 'v_ma20'] = new_data.ix[0, "volume"]
    for i in range(1, len(new_data.index)):
        new_data.ix[i, "price_change"] = new_data.ix[i, "close"] - new_data.ix[i-1, "close"]
        new_data.ix[i, "p_change"] = new_data.ix[i, "price_change"] / new_data.ix[i-1, "close"]
        previous = 0
        v_previous = 0
        for j in range(min(i+1, 5)):
            previous += new_data.ix[i - j, "close"]
            v_previous += new_data.ix[i - j, "volume"]
        new_data.ix[i, "ma5"] = previous / min(i+1, 5)
        new_data.ix[i, "v_ma5"] = v_previous / min(i+1, 5)
        for j in range(min(i+1, 5), min(i+1, 10)):
            previous += new_data.ix[i - j, "close"]
            v_previous += new_data.ix[i - j, "volume"]
        new_data.ix[i, "ma10"] = previous / min(i+1, 10)
        new_data.ix[i, "v_ma10"] = v_previous / min(i+1, 10)
        for j in range(min(i+1, 10), min(i+1, 20)):
            previous += new_data.ix[i - j, "close"]
            v_previous += new_data.ix[i - j, "volume"]
        new_data.ix[i, "ma20"] = previous / min(i+1, 20)
        new_data.ix[i, "v_ma20"] = v_previous / min(i+1, 20)
    return new_data
    
def CreateModel(batch_size, time_step, labels_len):
    model = Sequential()
    model.add(LSTM(20, batch_input_shape=(batch_size, time_step, labels_len), return_sequences=True, activation="softsign"))
    model.add(LSTM(20, return_sequences=True, activation="softsign"))
    model.add(LSTM(20, return_sequences=True, activation="softsign"))
    model.add(LSTM(20, return_sequences=True, activation="softsign"))
    model.add(LSTM(20, return_sequences=True, activation="softsign"))
    model.add(LSTM(20, return_sequences=True, activation="softsign"))
    model.add(LSTM(3, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["accuracy"])
    return model

def trans(origin):
    the_max = max(origin)
    result = []
    for value in origin:
        if value < the_max:
            result.append(0)
        else:
            result.append(1)
    return numpy.array(result)

def result_display(dates, signals):
    index = 0
    for signal in signals:
        if (signal == [1, 0, 0]).all():
            print(dates[index], "up")
        elif (signal == [0, 1, 0]).all():
            print(dates[index], "down")
        else :
            print(dates[index], "hold")
        index += 1

def proba_display(dates, signals):
    index = 0
    sign = ["up", "down", "still"]
    for signal in signals:
        [i], = numpy.where(signal == max(signal))
        print(dates[index], "up: ", signal[0]*100, "\tdown: ", signal[1]*100, "\tstill: ", signal[2]*100,
            "\t\t", sign[i])
        index += 1