import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset/nsl_kdd.csv")

print("Dataset Shape:", data.shape)
print(data.head())

data['label'] = data['label'].apply(lambda x: 'attack' if x != 'normal' else 'normal')

categorical_cols = data.select_dtypes(include=['object']).columns

encoder = LabelEncoder()
for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])

X = data.drop('label', axis=1)
y = data['label']

y = encoder.fit_transform(y)  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Attack'],
            yticklabels=['Normal', 'Attack'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - IDS")
plt.show()

sample_input = X_test[0].reshape(1, -1)
sample_prediction = model.predict(sample_input)

if sample_prediction[0] == 0:
    print("\nSample Result: ✅ Normal Traffic")
else:
    print("\nSample Result: 🚨 Intrusion Detected")