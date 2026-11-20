
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Engine_Size': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    'Mileage': [20, 18, 15, 14, 12, 11, 10, 9, 8, 7],
    'Age': [2, 3, 4, 5, 6, 7, 3, 2, 5, 4],
    'Price': [500000, 450000, 400000, 350000, 300000,
              280000, 600000, 700000, 550000, 650000]
}

df = pd.DataFrame(data)

# Split input and output
X = df[['Engine_Size', 'Mileage', 'Age']]
y = df['Price']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict car prices
y_pred = model.predict(X_test)

# Display accuracy
print("Car Price Prediction Model")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict price of a new car
new_car = pd.DataFrame({
    'Engine_Size': [1500],
    'Mileage': [15],
    'Age': [3]
})

predicted_price = model.predict(new_car)

print("Predicted Car Price: Rs.", round(predicted_price[0], 2))
