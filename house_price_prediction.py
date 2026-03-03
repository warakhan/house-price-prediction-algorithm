# STEP 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# STEP 2: Load dataset
data = pd.read_csv("housing.csv")

# STEP 3: Display first 5 rows
print(data.head())

# STEP 4: Check missing values
print(data.isnull().sum())

# STEP 5: Select input features and output
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# STEP 6: Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 7: Create Linear Regression model
model = LinearRegression()

# STEP 8: Train the model
model.fit(X_train, y_train)

# STEP 9: Predict house prices
y_pred = model.predict(X_test)

# STEP 10: Model accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy (R2 Score):", accuracy)

# STEP 11: Visualize results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# STEP 12: Predict price for new house
# Example: area=2000 sq ft, bedrooms=3, bathrooms=2
new_house = np.array([[2000, 3, 2]])
predicted_price = model.predict(new_house)

print("Predicted Price for new house:", predicted_price[0])
