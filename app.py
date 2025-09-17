import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("best_model.pkl")

st.title("🏠 House Price Prediction")
st.write("Predict house prices based on features")

# Sidebar or main input
mode = st.radio("Choose prediction mode:", ["Single House", "Multiple Houses (CSV)"])

if mode == "Single House":
    sqft = st.number_input("Square Feet", min_value=100, max_value=10000, value=1500)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    
    if st.button("Predict Price"):
        input_df = pd.DataFrame([{"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms}])
        price = model.predict(input_df)[0]
        st.success(f"Predicted Price: ₹{price:,.2f}")

else:  # CSV mode
    uploaded_file = st.file_uploader("Upload CSV with columns: sqft, bedrooms, bathrooms", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Keep only required columns
        df = df[["sqft", "bedrooms", "bathrooms"]]
        df["Predicted_Price"] = model.predict(df)
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")
