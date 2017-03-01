# StockPricePredict

这是一个尝试使用RNN预测股票市场的项目。项目的构思来源[这里](https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02#.i3v36u3fd)
数据源来自[Tushare](http://www.tushare.org/)
模型构建使用Keras

## 现状
在试验过程中发现，采用对每支股票单独训练一个模型进行预测的方案，比将所有股票数据合并来训练一个模型的方案，效果更好。
训练RNN模型时，发现activation使用softsign，速度更快。

## 接下来要完成的
1. 寻找更合适的ydata
2. 将公司的基本面数据加入模型中进行训练
