import joblib
import pandas as pd
import os

# Load the saved model
model = joblib.load("best_model.pkl")
print("✅ Model loaded successfully!\n")

# Ask user for CSV path
csv_path = input("Enter the path to the CSV file with house features: ")

if not os.path.exists(csv_path):
    print(f"❌ File not found: {csv_path}")
    exit()

# Load new data
new_data = pd.read_csv(csv_path)

# Check that required columns exist
required_cols = ["sqft", "bedrooms", "bathrooms"]
for col in required_cols:
    if col not in new_data.columns:
        print(f"❌ Missing column in CSV: {col}")
        exit()

# Predict
predictions = model.predict(new_data)

# Add predictions to DataFrame
new_data["Predicted_Price"] = predictions

# Show results
print("\n🏠 Predictions:")
print(new_data)

# Optional: save predictions to a new CSV
output_csv = "predictions.csv"
new_data.to_csv(output_csv, index=False)
print(f"\n✅ Predictions saved to {output_csv}")
