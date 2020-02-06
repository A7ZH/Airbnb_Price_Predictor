import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

train = pd.read_csv("data_train.csv", index_col=0)
test = pd.read_csv("data_test.csv", index_col=0)
train_y = np.log(train["price"].values)
test_y = np.log(test["price"].values)
train_X = train.drop(columns=["price", "new_factor"])
test_X = test.drop(columns=["price"])
assert (train_X.columns==test_X.columns).all(), "Features Unmatched in Train & Test Data"
train_X["neighbourhood_group"]=pd.factorize(train_X["neighbourhood_group"],sort=True)[0]+1
test_X["neighbourhood_group"]=pd.factorize(test_X["neighbourhood_group"],sort=True)[0]+1
train_X["room_type"] = pd.factorize(train_X["room_type"],sort=True)[0]+1
test_X["room_type"] = pd.factorize(test_X["room_type"],sort=True)[0]+1
train_X = train_X.values
test_X = test_X.values
train_X = (train_X - np.mean(train_X,axis=0))/np.std(train_X,axis=0) # Standardization
test_X = (test_X - np.mean(test_X,axis=0))/np.std(test_X,axis=0) # Standardization
numFeatures = train_X.shape[1]

model = Sequential()
model.add(Dense(units=256,activation="relu", input_shape=(numFeatures,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=64,activation="relu"))
model.add(Dense(units=32,activation="relu"))
model.add(Dense(units=16,activation="relu"))
model.add(Dense(units=1))
model.compile(optimizer="rmsprop", loss="mse", metrics= ["mean_absolute_error"])

history = model.fit(train_X,train_y,batch_size=32,epochs=40,shuffle=True,verbose=1,validation_split=0.0)

test_mse, test_mae = model.evaluate(test_X, test_y)

print("Test Mean Squared Error: " + str(test_mse))
print("Test Mean Absolute Error: " + str(test_mae))
print("Predicts:")
print(model.predict(test_X))
print(test_y)
'''
plt.subplot(2,2,1)
plt.plot(history.history["loss"])
plt.title("Training Mean Squared Error VS Iteration")
plt.ylabel("Training Mean Squared Error")
plt.xlabel("Iterations")

plt.subplot(2,2,2)
plt.plot(history.history["val_loss"])
plt.title("Validation Mean Squared Error VS Iteration")
plt.ylabel("Validation Mean Squared Error")
plt.xlabel("Iterations")

plt.subplot(2,2,3)
plt.plot(history.history["mean_absolute_error"])
plt.title("Training Mean Absolute Error VS Iteration")
plt.ylabel("Training Mean Absolute Error")
plt.xlabel("Iterations")

plt.subplot(2,2,4)
plt.plot(history.history["val_mean_absolute_error"])
plt.title("Validation Mean Absolute Error VS Iteration")
plt.ylabel("Validation Mean Absolute Error")
plt.xlabel("Iterations")

plt.plot()
'''
