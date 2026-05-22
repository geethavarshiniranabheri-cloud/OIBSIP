import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['message_length'] = df['message'].apply(len)

plt.figure(figsize=(6,5))

df['label'].value_counts().plot(kind='bar')

plt.xticks([0,1], ['Ham', 'Spam'])

plt.title("Spam vs Ham Messages")

plt.xlabel("Message Type")

plt.ylabel("Count")

plt.savefig("bar_graph.png")

plt.close()

plt.figure(figsize=(6,6))

df['label'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)

plt.title("Spam vs Ham Percentage")

plt.ylabel("")

plt.savefig("pie_chart.png")

plt.close()

plt.figure(figsize=(8,5))

plt.hist(
    df[df['label'] == 0]['message_length'],
    alpha=0.5,
    label='Ham'
)

plt.hist(
    df[df['label'] == 1]['message_length'],
    alpha=0.5,
    label='Spam'
)

plt.legend()

plt.title("Message Length Distribution")

plt.xlabel("Message Length")

plt.ylabel("Frequency")

plt.savefig("histogram.png")

plt.close()

x = df['message']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

model = MultinomialNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")

plt.close()

msg = ["Congratulations! You won a free iPhone"]

msg_data = cv.transform(msg)

prediction = model.predict(msg_data)

print("\nMessage:", msg[0])

if prediction[0] == 1:
    print("Prediction: Spam Message")
else:
    print("Prediction: Ham Message")