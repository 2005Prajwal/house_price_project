import joblib
import pandas as pd
import os

# Load the saved model
model = joblib.load("best_model.pkl")
print("✅ Model loaded successfully!\n")

# Choose mode
print("Choose prediction mode:")
print("1: Single house")
print("2: Multiple houses from CSV")
choice = input("Enter 1 or 2: ").strip()

feature_cols = ["sqft", "bedrooms", "bathrooms"]

if choice == "1":
    # Single house
    try:
        sqft = float(input("Enter square feet: "))
        bedrooms = int(input("Enter number of bedrooms: "))
        bathrooms = int(input("Enter number of bathrooms: "))
    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")
        exit()

    new_house = pd.DataFrame([{
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    }])

    predicted_price = model.predict(new_house)[0]
    print(f"\n🏠 Predicted House Price: {predicted_price:.2f}")

elif choice == "2":
    # Multiple houses from CSV
    csv_path = input("Enter the path to the CSV file with house features: ").strip()
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        exit()

    new_data = pd.read_csv(csv_path)

    # Keep only required columns
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        print(f"❌ Missing required columns in CSV: {missing_cols}")
        exit()

    X_new = new_data[feature_cols]

    predictions = model.predict(X_new)
    new_data["Predicted_Price"] = predictions

    print("\n🏠 Predictions:")
    print(new_data)

    output_csv = "predictions.csv"
    new_data.to_csv(output_csv, index=False)
    print(f"\n✅ Predictions saved to {output_csv}")

else:
    print("❌ Invalid choice! Enter 1 or 2.")

