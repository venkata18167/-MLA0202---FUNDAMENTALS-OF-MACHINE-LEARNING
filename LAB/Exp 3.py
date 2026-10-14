# ID3 Decision Tree Algorithm
# Using Entropy and Information Gain

import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# 1. Create the dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
                'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
                'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool',
                    'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
                    'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal',
                 'Normal', 'Normal', 'High', 'Normal', 'Normal',
                 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
             'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes',
                   'No', 'Yes', 'No', 'Yes', 'Yes',
                   'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

print("Training Dataset:")
display(df)

# 2. Encode categorical values
X = df.drop('PlayTennis', axis=1)
y = df['PlayTennis']

encoders = {}

X_encoded = X.copy()

for col in X.columns:
    encoders[col] = LabelEncoder()
    X_encoded[col] = encoders[col].fit_transform(X[col])

target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# 3. Build the decision tree using entropy (ID3 criterion)
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X_encoded, y_encoded)

# 4. Visualize the decision tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=target_encoder.classes_,
    filled=True,
    rounded=True
)

plt.title("ID3 Decision Tree - Play Tennis")
plt.show()

# 5. Classify a new sample
new_sample = pd.DataFrame([{
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'Normal',
    'Wind': 'Weak'
}])

# Encode the sample using the training encoders
new_encoded = new_sample.copy()

for col in X.columns:
    new_encoded[col] = encoders[col].transform(new_sample[col])

prediction = model.predict(new_encoded)

predicted_class = target_encoder.inverse_transform(prediction)

print("\nNew Sample:")
display(new_sample)

print("Predicted Class:", predicted_class[0])
