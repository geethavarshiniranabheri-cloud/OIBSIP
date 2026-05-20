import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("car_prediction_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='Fuel_Type', data=df)
plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.savefig("fuel_type_distribution.png")
plt.close()

plt.figure(figsize=(6,4))
sns.countplot(x='Transmission', data=df)
plt.title("Transmission Distribution")
plt.xlabel("Transmission")
plt.ylabel("Count")
plt.savefig("transmission_distribution.png")
plt.close()

plt.figure(figsize=(8,5))
sns.lineplot(x='Year', y='Selling_Price', data=df)
plt.title("Year vs Selling Price")
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.savefig("year_vs_price.png")
plt.close()

df.replace({
    'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2},
    'Seller_Type': {'Dealer':0, 'Individual':1},
    'Transmission': {'Manual':0, 'Automatic':1}
}, inplace=True)

plt.figure(figsize=(10,8))
numeric_df = df.drop(['Car_Name'], axis=1)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

X = df.drop(['Car_Name', 'Selling_Price'], axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

plt.figure(figsize=(8,6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Graphs saved successfully")