import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

data=pd.read_csv("data/processed_dataset.csv")
x=data.iloc[:,:-1]
y=data['ElectricityBill']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
LR=LinearRegression()
LR.fit(x_train,y_train)
y_pred=LR.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=mse**0.5
print("Mean absolute error =",mae)
print("Mean Squared error =",mse)
print("RMSE =",rmse)
print("r2 Square =",r2)
