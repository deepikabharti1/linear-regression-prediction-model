import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df=pd.read_csv("housing.csv")
print(df.isnull().sum())
print(df.dtypes)
df1=df.drop("id",axis=1)

df_encoded=pd.get_dummies(df1,columns=["area"],dtype=int) #encoding
print('\n df one hot encoded data')
print(df_encoded.head(5))

x=df_encoded.drop("price_inr", axis=1)
y=df_encoded["price_inr"]

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42) #splitting
print(x_train.shape)
print(x_test.shape)

scaler= StandardScaler()
x_train_scaled=scaler.fit_transform(x_train) #scaling
x_test_scaled=scaler.transform(x_test)

model=LinearRegression()
model.fit(x_train_scaled,y_train) #fitting

y_pred=model.predict(x_test_scaled) #prediction
model_prediction=pd.DataFrame({'actual': y_test,
                     'predicted': y_pred}).round(2)

print(model_prediction)

mae=mean_absolute_error(y_test, y_pred) #metrics
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
print("MAE:" ,round(mae,2))
print("MSE:" , round(mse,2))
print("r2_score:",round(r2,2))

y.mean() #avg price

print('avg_price:' ,  round(y.mean(), 2)) # compare mae with avg

plt.scatter(y_test, y_pred) #visualization
plt.xlabel('test samples')
plt.ylabel('house price')
plt.title('actual vs predicted house prices')
plt.show()

print(model.coef_ .round(2))
print(model.intercept_.round(2))
