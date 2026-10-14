# ============================================
# Candidate Elimination Algorithm
# ============================================

import pandas as pd
import itertools
import matplotlib.pyplot as plt
from google.colab import files

# --------------------------------------------
# 1. Upload CSV file
# --------------------------------------------

uploaded = files.upload()

filename = list(uploaded.keys())[0]

data = pd.read_csv(filename)

print("Training Dataset:")
display(data)

# --------------------------------------------
# 2. Separate attributes and target
# --------------------------------------------

attributes = list(data.columns[:-1])
target = data.columns[-1]

print("\nAttributes:", attributes)
print("Target:", target)

# --------------------------------------------
# 3. Initialize S and G
# --------------------------------------------

S = ['∅'] * len(attributes)

G = [['?'] * len(attributes)]

print("\nInitial Specific Boundary (S):")
print(S)

print("\nInitial General Boundary (G):")
print(G)


# --------------------------------------------
# 4. Candidate Elimination Algorithm
# --------------------------------------------

for index, row in data.iterrows():

    example = list(row[attributes])
    label = row[target]

    print("\n--------------------------------------")
    print("Training Example", index + 1)
    print("--------------------------------------")
    print(example, "=>", label)

    # ----------------------------------------
    # Positive Example
    # ----------------------------------------
    if str(label).lower() == 'yes':

        # Remove inconsistent hypotheses from G
        G = [
            g for g in G
            if all(g[i] == '?' or g[i] == example[i] for i in range(len(attributes)))
        ]

        # Generalize S
        for i in range(len(attributes)):

            if S[i] == '∅':
                S[i] = example[i]

            elif S[i] != example[i]:
                S[i] = '?'

        # Remove hypotheses from G that are not more general than S
        G = [
            g for g in G
            if all(
                S[i] == '?' or
                g[i] == '?' or
                g[i] == S[i]
                for i in range(len(attributes))
            )
        ]

    # ----------------------------------------
    # Negative Example
    # ----------------------------------------
    else:

        # Remove inconsistent hypotheses from S
        if all(
            S[i] == '?' or S[i] == example[i]
            for i in range(len(attributes))
        ):
            # S is not changed directly in the simplified implementation
            pass

        new_G = []

        for g in G:

            # If G hypothesis covers negative example,
            # specialize it.
            if all(
                g[i] == '?' or g[i] == example[i]
                for i in range(len(attributes))
            ):

                for i in range(len(attributes)):

                    if g[i] == '?':
                        if S[i] != '?' and S[i] != '∅':
                            new_h = g.copy()
                            new_h[i] = S[i]

                            if new_h not in new_G:
                                new_G.append(new_h)
            else:
                new_G.append(g)

        G = new_G

    print("S =", S)
    print("G =", G)


# --------------------------------------------
# 5. Remove duplicate hypotheses
# --------------------------------------------

G_unique = []

for g in G:
    if g not in G_unique:
        G_unique.append(g)

G = G_unique


# --------------------------------------------
# 6. Display Final Version Space
# --------------------------------------------

print("\n\n============================================")
print("FINAL RESULT")
print("============================================")

print("\nSpecific Boundary (S):")
print(S)

print("\nGeneral Boundary (G):")

for g in G:
    print(g)

print("\n============================================")
print("All hypotheses consistent with training data")
print("============================================")


# --------------------------------------------
# 7. Generate hypotheses between S and G
# --------------------------------------------

domain_values = []

for attribute in attributes:
    domain_values.append(
        list(data[attribute].unique()) + ['?']
    )

all_hypotheses = list(itertools.product(*domain_values))

consistent_hypotheses = []

for h in all_hypotheses:

    consistent = True

    for _, row in data.iterrows():

        example = list(row[attributes])
        label = str(row[target]).lower()

        covers = all(
            h[i] == '?' or h[i] == example[i]
            for i in range(len(attributes))
        )

        if label == 'yes' and not covers:
            consistent = False
            break

        if label == 'no' and covers:
            consistent = False
            break

    if consistent:
        consistent_hypotheses.append(h)


print("\nNumber of Consistent Hypotheses:",
      len(consistent_hypotheses))

for h in consistent_hypotheses:
    print(h)


# --------------------------------------------
# 8. Visualization
# --------------------------------------------

plt.figure(figsize=(12, 5))

x = range(len(attributes))

plt.plot(
    x,
    [1 if s != '∅' else 0 for s in S],
    marker='o',
    label='Specific Boundary (S)'
)

plt.xticks(x, attributes)
plt.ylabel("Hypothesis Status")
plt.xlabel("Attributes")
plt.title("Candidate-Elimination Algorithm")
plt.grid(True)
plt.legend()

plt.show()
