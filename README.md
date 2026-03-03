🏠 House Price Prediction System

A Machine Learning-based web application that predicts house prices using property features such as area, number of bedrooms, and bathrooms. The model is built using Linear Regression and deployed using Streamlit for real-time predictions.

## 🚀 Project Overview

This project implements an end-to-end Machine Learning workflow:
- Data preprocessing and feature selection
- Train-test split for model validation
- Linear Regression model training
- Model evaluation using R² Score
- Deployment as an interactive Streamlit web application

Users can input property details and instantly receive a predicted house price along with model performance insights.
  ## 🧠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
 ## 📊 Machine Learning Process

1. Load housing dataset (CSV file)
2. Select relevant features (area, bedrooms, bathrooms)
3. Split dataset into training and testing sets
4. Train Linear Regression model
5. Evaluate using R² Score
6. Visualize Actual vs Predicted Prices
7. Deploy model using Streamlit
##  💻 How to Run the Project
Step 1: Clone Repository

```bash
git clone https://github.com/your-username/House_Price_Prediction.git
cd House_Price_Prediction
Step 2: Install Dependencies
pip install pandas numpy scikit-learn matplotlib streamlit
Step 3: Run the Web App
python -m streamlit run app.py
Open in browser:

http://localhost:8501
## 📈 Model Details
Algorithm: Linear Regression
Evaluation Metric: R² Score
Output: Predicted house price with visualization

## ⚠️Limitations
The dataset does not include location information. Predictions are based solely on structural property features.

## 🎯 Future Enhancements
Add location-based pricing
Implement Random Forest for improved accuracy
Deploy on Streamlit Cloud
Add downloadable prediction reports
