Link:https://2005prajwal-house-price-project-app-pifib5.streamlit.app/


🏡 House Price Prediction (Regression Models)

This project applies machine learning regression models to predict house prices based on features such as square footage, number of bedrooms, and bathrooms.

🚀 Features

Trains multiple regression models (Linear Regression, Decision Tree, Random Forest, etc.)

Compares model performance using R² and RMSE

Saves the best model as best_model.pkl

Predicts house prices for:

Single House (via script or Streamlit app)

Multiple Houses (CSV upload)

Interactive Streamlit app for user input and visualization

📂 Project Structure

house_prices.csv → Dataset

task1.py → Training & evaluation script

predict.py → Predict for a single house

predict_multiple.py → Predict for multiple houses (CSV input)

app.py → Streamlit app

best_model.pkl → Saved trained model

requirements.txt → Dependencies

▶️ Run Locally
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate   # (Windows)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save best model
python task1.py

# 4. Predict single house price
python predict.py

# 5. Predict multiple houses (CSV)
python predict_multiple.py

# 6. Run web app
streamlit run app.py
