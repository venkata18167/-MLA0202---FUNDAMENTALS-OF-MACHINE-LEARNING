
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Income': [30000, 50000, 70000, 25000, 80000, 45000, 90000, 35000],
    'Debt': [20000, 15000, 10000, 25000, 5000, 18000, 3000, 22000],
    'Payment_History': [2, 7, 9, 1, 10, 5, 10, 2],
    'Credit_Score': ['Poor', 'Standard', 'Good', 'Poor',
                     'Good', 'Standard', 'Good', 'Poor']
}

df = pd.DataFrame(data)

# Split input and output
X = df[['Income', 'Debt', 'Payment_History']]
y = df['Credit_Score']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict credit score
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")

# Predict for a new customer
new_customer = pd.DataFrame(
    [[60000, 12000, 8]],
    columns=['Income', 'Debt', 'Payment_History']
)

prediction = model.predict(new_customer)
print("Predicted Credit Score:", prediction[0])
