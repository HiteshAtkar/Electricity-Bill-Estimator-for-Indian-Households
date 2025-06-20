import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pickle
import os


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
mlflow.set_experiment("LR Test 1")
with mlflow.start_run(run_name="Test2"):
    mlflow.log_metrics({
        "Mean absolute error":mae,
        "Mean Squared error":mse,
        "r2":r2
    })
    input_example=x_test.iloc[:2]
    signature=infer_signature(x_test,y_pred)
    mlflow.sklearn.log_model(sk_model=LR,name="Linear Regression model",input_example=input_example,signature=signature)

os.mkdirs("model",exist_ok=True)
with open('model/model.pkl','wb') as f:
    pickle.dump(LR,f)
    print("_________Model Successfully Saved________")

print("Mean absolute error =",mae)
print("Mean Squared error =",mse)
print("RMSE =",rmse)
print("r2 Square =",r2)
