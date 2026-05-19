import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Unemployment in India.csv")

# Display first 5 rows
print(df.head())

# Dataset information
print(df.info())

# Check null values
print(df.isnull().sum())

# Graph
plt.figure(figsize=(12,6))

sns.barplot(
    x='Region',
    y=' Estimated Unemployment Rate (%)',
    data=df
)

plt.xticks(rotation=90)

plt.title("Unemployment Rate by Region")

plt.show()