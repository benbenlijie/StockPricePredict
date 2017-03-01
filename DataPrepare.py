import numpy
import pandas
import math

from sklearn.preprocessing import MinMaxScaler

class DataPrepare:
    
    def __init__(self, dataFrame, labels=None):
        self.origin_data = dataFrame
        self.time_step = 20
        self.forecast_time = 10
        self.rate = 0.65
        if labels is None:
            labels = list(dataFrame.columns)
        print(labels)
        self.labels = labels
        self.scaler = dict()
        for label in self.labels:
            self.scaler[label] = SerialDataScaler(dataFrame.ix[:, label])
        self.seperate(dataFrame)
        
            
    def seperate(self, data_arr):
        train_size = int(len(data_arr) * self.rate)
        test_size = len(data_arr) - train_size
        self.train, self.test = data_arr.ix[0:train_size, :], data_arr.ix[train_size:len(data_arr), :]
        
    def create_dataset(self, data_sets):
        dataX, dataY = [], []
        scaled_data_sets = {}
        for label in self.labels:
            scaled_data_sets[label] = self.scaler[label].transform(data_sets.ix[:, label])
        y_data = data_sets["close"]
        date = []
        for i in range(len(data_sets) - self.time_step - 1 - self.forecast_time):
            x = []
            for j in range(self.time_step):
                time_slice = []
                for key in self.labels:
                    time_slice.append(scaled_data_sets[key][i+j])
                x.append(time_slice)
            dataX.append(x)
            index = i + self.time_step
            date.append(index)
            dataY.append(self.calculate_y(y_data, index, period=self.forecast_time))
        dataX = numpy.array(dataX)
        dataX = numpy.reshape(dataX, (dataX.shape[0], dataX.shape[1], dataX.shape[2]))
        return dataX, numpy.reshape(numpy.array(dataY), (len(dataY), 3)), date

    def calculate_y(self, data, index, period=5):
        future = 0
        cur = 0
        for i in range(period):
            future += data.ix[index+i]
            cur += data.ix[index-i-1]
        cur /= period
        future /= period
        rate = future / cur - 1.0
        if (rate) > 0.02:
            return [1, 0, 0]
        elif (rate < 0):
            return [0, 1, 0]
        else:
            return [0, 0, 1]
        
            
class SerialDataScaler:
    
    def __init__(self, data):
        data = numpy.reshape(data, (len(data), 1))
        data = data.astype("float32")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(data)
    
    def transform(self, X):
        #return X
        return self.scaler.transform(numpy.reshape(X, (len(X), 1)))

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)
