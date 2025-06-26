# Flat Price Estimator - User-Friendly, GitHub-Ready Version

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from google.colab import files
    COLAB = True
except ImportError:
    COLAB = False


def train_model():
    print("\n📁 Upload your training dataset (must include 'flat_price' column)")
    if COLAB:
        uploaded = files.upload()
        dataset_name = list(uploaded.keys())[0]
    else:
        dataset_name = input("Enter dataset filename: ")

    df = pd.read_csv(dataset_name)
    X = df.drop("flat_price", axis=1)
    y = df["flat_price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n📊 Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: ₹{mae:,.2f}")
    print(f"RMSE: ₹{rmse:,.2f}")

    model_filename = "flat_price_model.pkl"
    joblib.dump(model, model_filename)
    print(f"\n✅ Model saved as {model_filename}")
    if COLAB:
        files.download(model_filename)
    return model


def load_model():
    print("\n📁 Upload your trained model file (.pkl)")
    if COLAB:
        uploaded = files.upload()
        model_filename = list(uploaded.keys())[0]
    else:
        model_filename = input("Enter model filename: ")
    model = joblib.load(model_filename)
    print(f"✅ Model loaded from {model_filename}")
    return model


def predict_manual(model):
    print("\n✍️ Enter flat details:")
    try:
        area = float(input("Area in sqft: "))
        bedrooms = int(input("Number of bedrooms: "))
        distance = float(input("Distance to metro in km: "))
        age = int(input("Age of flat in years: "))
        amenities = float(input("Amenities score (0-10): "))
    except ValueError:
        print("❌ Invalid input. Please enter numeric values only.")
        return

    user_data = np.array([[area, bedrooms, distance, age, amenities]])
    predicted_price = model.predict(user_data)[0]
    print(f"\n💰 Estimated Flat Price: ₹{predicted_price:,.2f}")


def predict_bulk(model):
    print("\n📁 Upload CSV file with flat details")
    if COLAB:
        uploaded_test = files.upload()
        test_file = list(uploaded_test.keys())[0]
    else:
        test_file = input("Enter CSV file name: ")

    test_df = pd.read_csv(test_file)
    predictions = model.predict(test_df)
    test_df["predicted_price"] = predictions

    print("\n📄 Bulk Predictions:")
    print(test_df)

    output_file = "bulk_predictions_output.csv"
    test_df.to_csv(output_file, index=False)
    print(f"✅ Results saved to {output_file}")
    if COLAB:
        files.download(output_file)


# ==== Main Program Entry ====
print("\n🏡 Welcome to Flat Price Estimator")
print("1. Train a new model")
print("2. Load existing model")
choice = input("Enter 1 or 2: ")

model = None
if choice.strip() == "1":
    model = train_model()
elif choice.strip() == "2":
    model = load_model()
else:
    print("❌ Invalid selection. Exiting.")
    exit()

print("\n🔍 Choose Prediction Mode:")
print("1. Manual input")
print("2. Bulk prediction from CSV")
predict_choice = input("Enter 1 or 2: ")

if predict_choice.strip() == "1":
    predict_manual(model)
elif predict_choice.strip() == "2":
    predict_bulk(model)
else:
    print("❌ Invalid prediction option. Exiting.")
