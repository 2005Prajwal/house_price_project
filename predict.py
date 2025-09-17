# predict.py (interactive version)
import joblib
import pandas as pd

# Load the saved model
model = joblib.load("best_model.pkl")
print("✅ Model loaded successfully!\n")

# Ask user for input
try:
    sqft = float(input("Enter square feet: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    bathrooms = int(input("Enter number of bathrooms: "))
except ValueError:
    print("❌ Invalid input! Please enter numbers only.")
    exit()

# Create a DataFrame for the input
new_house = pd.DataFrame([{
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms
}])

# Predict
predicted_price = model.predict(new_house)[0]

# Show result
print(f"\n🏠 Predicted House Price: {predicted_price:.2f}")
