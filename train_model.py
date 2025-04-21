# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Manually entering the data from the Excel file
data = {
    "GENDER": ['F', 'M', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'M'],
    "AGE": [65, 61, 54, 49, 45, 36, 60, 63, 45, 52],
    "SMOKING": [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
    "YELLOW_FINGERS": [1, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    "ANXIETY": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    "PEER_PRESSURE": [1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    "CHRONIC DISEASE": [1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    "FATIGUE ": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    "ALLERGY ": [0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    "WHEEZING": [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    "ALCOHOL CONSUMING": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    "COUGHING": [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    "SHORTNESS OF BREATH": [1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
    "SWALLOWING DIFFICULTY": [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
    "CHEST PAIN": [1, 1, 1, 1, 0, 1, 0, 0, 1, 1],
    "LUNG_CANCER": ['YES', 'YES', 'YES', 'NO', 'NO', 'NO', 'NO', 'NO', 'NO', 'YES']
}

df = pd.DataFrame(data)

# Encode categorical features
le_gender = LabelEncoder()
df['GENDER'] = le_gender.fit_transform(df['GENDER'])
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump((model, le_gender), "cancer_model.pkl")
print("Model trained and saved to cancer_model.pkl")
