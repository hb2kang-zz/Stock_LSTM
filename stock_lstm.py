import sys
import os
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

#Retrieves stock data from alphavantage
def get_stock_data(stock_name, api_key, output_size):
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&apikey=%s&datatype=csv&outputsize=%s' % (stock_name, api_key, output_size)
    #load csv data from url
    stocks = pd.read_csv(url)
    #load data to DataFrame
    df = pd.DataFrame(stocks)
    
    return df

#Function to seperate Training data and Testing data

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

#LSTM

def lstm_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop",metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model

#seq2seq model

def seq2seq(layers):
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16, activation='relu'))        
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])
        return model


def main():

	#Get Data

	api_key = '5HSLKBUWSP7B6DOX'
	stock_name = 'MSFT'

	output_size = 'full'
	df = get_stock_data(stock_name, api_key, output_size)

	print(df.head(10))

	print("The shape of dataframe is:")

	print(df.shape)

	#Feature selection and process

	data = df.drop(df.columns[[0,3,5]], axis=1)

	print(data.head(5))

	#Data to csv

	today = datetime.date.today()

	dir_path = os.path.dirname(os.path.realpath('stock_lstm.py'))
	file_name = dir_path + "/data/" + stock_name +'_stock_%s.csv' % today
	data.to_csv(file_name)

	print(data)

	#Feauture Scaling

	data['high'] = data['high'] / 100
	data['open'] = data['open'] / 100
	data['close'] = data['close'] / 100
	print(data.head(5))

	#Separate Data

	#Chosen the days when exchange is operating
	seq_length = 22
	X_train, y_train, X_test, y_test = load_data(data[::-1], seq_length)
	print("X_train:", X_train.shape)
	print("y_train:", y_train.shape)
	print("X_test:", X_test.shape)
	print("y_test:", y_test.shape)

	# model = build_model([3,lag,1])
	model = seq2seq([3, seq_length ,1])

	#Fit model and train

	model.fit(
    X_train,
    y_train,
    batch_size=256,
    nb_epoch=100,
    validation_split=0.1,
    verbose=1)


	#Score

	trainScore = model.evaluate(X_train, y_train, verbose=0)
	print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

	testScore = model.evaluate(X_test, y_test, verbose=0)
	print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

	#getPredictionData('MSFT',0)
	window = 22
	diff=[]
	ratio=[]
	p = model.predict(X_test)

	for u in range(0, len(X_test)):
		pr = p[u][0]
		ratio.append((y_test[u]/pr)-1)
		diff.append(abs(y_test[u]- pr))
    	#print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))


	import matplotlib.pyplot as plt2
	print("Expected value: ",(y_test[-1] * 100))
	print("Last Input: ",(X_test[-1] * 100))
	print("Last Prediction: ",(p[-1] * 100))
	plt2.plot(p ,color='red', label='prediction')
	plt2.plot(y_test ,color='blue', label='y_test')

	#plt2.legend(loc='upper left')
	plt2.show()

	print("The Stock values have been plotted.")

	sys.exit()

if __name__ == '__main__':
	main()
