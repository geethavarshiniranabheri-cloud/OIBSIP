import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Advertising.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

# Pairplot Graph
sns.pairplot(df)
plt.savefig("pairplot.png")
plt.close()

# Heatmap Graph
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="Blues")
plt.savefig("heatmap.png")
plt.close()

# TV vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['TV'], y=df['Sales'])
plt.title("TV Advertising vs Sales")
plt.savefig("tv_vs_sales.png")
plt.close()

# Radio vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Radio'], y=df['Sales'])
plt.title("Radio Advertising vs Sales")
plt.savefig("radio_vs_sales.png")
plt.close()

# Newspaper vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Newspaper'], y=df['Sales'])
plt.title("Newspaper Advertising vs Sales")
plt.savefig("newspaper_vs_sales.png")
plt.close()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredicted Sales:\n")
print(y_pred)

print("\nModel Evaluation\n")

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample_data = [[230.1, 37.8, 69.2]]

prediction = model.predict(sample_data)

print("\nPredicted Sales for sample data:", prediction[0])

print("\nGraphs saved successfully!")