#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib as mlp
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
Config.set('graphics', 'width', '1150')
Config.set('graphics', 'height', '600')
register_matplotlib_converters()
plt.style.use('fivethirtyeight')
pb = ProgressBar(max = 100)

#import webbrowser

df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-04-21')

class MainScreen(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def graph_display(self):
        plt.plot(df['Close'])
        plt.xlabel('Date')
        plt.ylabel('Close Price USD ($)')
        box = BoxLayout()
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        self.add_widget(box)
        

    def processing(self):
        data = df.filter(['Close'])
        dataset = data.values
        training_data_len = math.ceil( len(dataset) * .85 )
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        #<--------------!!!-------------->
        #Create the training data set
        #Create the scaled training data set
        train_data = scaled_data[0:training_data_len , :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if i<= 61:
                #print(x_train)
                #print(y_train)
                print()

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences= False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=1, epochs = 1)
 
        test_data = scaled_data[training_data_len - 60: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
        print(rmse)
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        box = BoxLayout()
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        self.add_widget(box)

        apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2020-03-26')
        new_df = apple_quote.filter(['Close'])
        last_60_days = new_df[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        print(pred_price)

        apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2020-03-27', end='2020-03-27')
        print(apple_quote2['Close'])


class MyApp(App):
    title = 'Price Prediction' 
    def build(self):
        return MainScreen()

application = MyApp()
application.run()

#if __name__ == "__main__":
  #plot()
  #processing()
  #plot_result()
  #show_prices()

