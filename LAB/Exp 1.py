import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# 1. Training Dataset
# -----------------------------------------
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'AirTemp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'EnjoySport': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

print("Training Dataset:")
display(df)

# -----------------------------------------
# 2. FIND-S Algorithm
# -----------------------------------------

# Initial hypothesis: most specific hypothesis
hypothesis = ['Ø'] * (len(df.columns) - 1)

print("\nInitial Hypothesis:")
print(hypothesis)

# Store hypotheses for visualization
hypothesis_history = [hypothesis.copy()]

# Process each training example
for index, row in df.iterrows():

    # Consider only positive examples
    if row['EnjoySport'] == 'Yes':

        for i in range(len(hypothesis)):

            # First positive example
            if hypothesis[i] == 'Ø':
                hypothesis[i] = row.iloc[i]

            # If attribute values are different
            elif hypothesis[i] != row.iloc[i]:
                hypothesis[i] = '?'

        hypothesis_history.append(hypothesis.copy())

        print(f"\nExample {index + 1}:")
        print(list(row))
        print("Hypothesis:", hypothesis)

# -----------------------------------------
# 3. Final Hypothesis
# -----------------------------------------

print("\n--------------------------------")
print("Final Most Specific Hypothesis:")
print("--------------------------------")
print(hypothesis)

# -----------------------------------------
# 4. Visualization using Matplotlib
# -----------------------------------------

attributes = df.columns[:-1]

fig, ax = plt.subplots(figsize=(12, 6))

for i, attr in enumerate(attributes):
    values = []

    for h in hypothesis_history:
        values.append(h[i])

    ax.plot(
        range(len(values)),
        values,
        marker='o',
        label=attr
    )

ax.set_title("FIND-S Algorithm - Hypothesis Learning Process")
ax.set_xlabel("Training Step")
ax.set_ylabel("Hypothesis Value")

ax.set_xticks(range(len(hypothesis_history)))
ax.set_xticklabels(
    ['Initial'] + [f'Step {i}' for i in range(1, len(hypothesis_history))]
)

ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()
