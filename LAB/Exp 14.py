
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 4, 5],
    'Age': [5, 4, 3, 6, 2, 3, 1, 2],
    'Price': [2500000, 3000000, 4000000, 4500000,
              5500000, 6000000, 7000000, 8000000]
}

df = pd.DataFrame(data)

# Split input and output
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Display results
print("House Price Prediction")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict a new house price
new_house = pd.DataFrame({
    'Area': [1600],
    'Bedrooms': [3],
    'Age': [2]
})

predicted_price = model.predict(new_house)

print("Predicted House Price: Rs.", round(predicted_price[0], 2))
