import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


def get_stock_data(stock_name, isTraining, normalized=0):
    url = 'http://chart.finance.yahoo.com/table.csv?s=%s&a=11&b=15&c=2011&d=29&e=10&f=2016&g=d&ignore=.csv' % stock_name
    col_names = ['Date','Open','High','Low','Close','Volume','Adj Close']
    stocks = pd.read_csv(url, header=0, names=col_names) 
    df = pd.DataFrame(stocks)
    #if not isTraining:
       # df1 = pd.DataFrame({'Date': ["2017-04-26"],
                   #       'Open': [312.37],
                         # 'High': [314.50],
                        #  'Low': [310.25],
                        #  'Close': [0],
                        #  'Volume': [0],
                        #  'Adj Close': [0]},
                        #  index= [0])
        #frames = [df1, df]
        #df = pd.concat(frames)
    date_split = df['Date'].str.split('-').str
    df['Year'], df['Month'], df['Day'] = date_split
    df["Volume"] = df["Volume"] / 10000
    date_store = df
 
    df.drop(df.columns[[0,3,5,6, 7,8,9]], axis=1, inplace=True)
    return df, date_store

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]




class stock_lstm():

	def __init__(self, layers):
		self.layers = layers

	def create_model(self, dropout):
		d = 0.2

		#create Sequential object from keras
		self = Sequential()
		self.add(LSTM(128, input_shape=(self.layers[1], self.layers[0]), return_sequences=True))
		self.add(Dropout(dropout))
		self.add(LSTM(64, input_shape=(self.layers[1], self.layers[0]), return_sequences=False))
		self.add(Dropout(dropout))
		self.add(Dense(16,kernel_initializer='uniform',activation='relu'))        
		self.add(Dense(1,kernel_initializer='uniform',activation='linear'))
		self.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

		return self


	def fit(self, x_train, y_train):
		self.fit(
    	x_train,
    	y_train,
    	batch_size=256,
    	nb_epoch=500,
    	validation_split=0.1,
    	verbose=1)

		return self

	def Score(self):

		trainScore = self.evaluate(x_train, y_train, verbose=0)
		print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
		testScore = self.evaluate(x_test, y_test, verbose=0)
		print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


def main():

	#select stock
	stock_name = 'TSLA'
	df, date_store = get_stock_data(stock_name, 0, True)
	print(df.head(23))

	#write csv
	today = datetime.date.today()
	fromile_name = stock_name+'_stock_%s.csv' % today
	df.to_csv(file_name)


	#feature scaling
	df['High'] = df['High'] / 100
	df['Open'] = df['Open'] / 100
	df['Close'] = df['Close'] / 100
	print(df.head(5))


	#load data to dataframe
	window = 22
	x_train, y_train, x_test, y_test = load_data(df[::-1], window)
	print("X_train", x_train.shape)
	print("y_train", y_train.shape)
	print("X_test", x_test.shape)
	print("y_test", y_test.shape)

	print(x_train)




if __name__ == '__main__':
	main()



