import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

np.random.seed(2019)
from tensorflow import set_random_seed
set_random_seed(2019)

"""Step 1: Read data from file and Preprocess"""
train_df = pd.read_csv('../data/dow_train_regression_change.csv')
train_df.head()

train_y = train_df['DJI_change']
train_x = train_df.drop(columns = ['Date', 'DJI_change'])

"""Step 2: Build Model"""
model = Sequential()
model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

"""Step 3: Train Model"""
model.fit(train_x, train_y, epochs=37)

"""Step 4: Load test data for prediction"""
test_df = pd.read_csv('../data/dow_test_regression_change.csv')
test_y = test_df['DJI_change']
test_x = test_df.drop(columns = ['Date', 'DJI_change'])
pred_y = model.predict(test_x)
print('MSE = {}'.format(mean_squared_error(test_y, pred_y)))
print('RMSE = {}'.format(mean_squared_error(test_y, pred_y) ** 0.5))
