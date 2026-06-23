#Decode Labs Internship - Week 2 Project
#Data Classification using K-Nearest Neighbors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , classification_report , accuracy_score 

# ==========================================================
# 1. Load and Explore the Iris Dataset
# ==========================================================
iris_dataset = load_iris()

iris_df = pd.DataFrame(data=iris_dataset.data,
                       columns=iris_dataset.feature_names)
iris_df["species"] = iris_dataset.target

# Uncomment to inspect the dataset
# print(iris_df.shape)
# print(iris_df.head())
# print(iris_dataset.feature_names)
# print(iris_dataset.target_names)
# print(iris_df["species"].value_counts())

# ==========================================================
# 2. Train_Test_split - 80/20 split
# ==========================================================

X, y = iris_df.drop("species", axis=1) , iris_df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
# print(X_train.shape,X_test.shape)
# print(y_train.shape,y_test.shape)

# ==========================================================
# 3. Feature Scaling
# ==========================================================
# KNN relies on distance calculations, so scaling is important.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# print(scaler.mean_)
# print(scaler.var_)

X_test_scaled = scaler.transform(X_test)
# print(X_train_scaled)
# print(X_test_scaled)

# ==========================================================
# 4. Find the Best K Value
# ==========================================================
accuracy = []
for K in range(1,21) :
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train_scaled,y_train)
    predictions = model.predict(X_test_scaled)
    accuracy.append(accuracy_score(y_test,predictions))

# argmax returns the index of highest value in accuracy list
# +1 is because our k value begin from 1 but accuracy list is 0-indexed
best_k = np.argmax(accuracy) + 1 
best_accuracy = accuracy[np.argmax(accuracy)]

print(f"Best K Value: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")


# ==========================================================
# 5. Plot K vs Accuracy
# ==========================================================
x_axis = range(1,21)
plt.plot(x_axis, accuracy, marker="o")
plt.grid(True)
plt.xlabel('K values')
plt.ylabel('Accuracy_score')
plt.title('K Vs accuracy plot - elbow curve')
plt.savefig("k_vs_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================================
# 6. Train Final Model Using Best K
# ==========================================================
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled,y_train)
predictions = model.predict(X_test_scaled)

# ==========================================================
# 7. Display Predictions
# ==========================================================
print("\nPredicted Classes:")
print(predictions)

# ==========================================================
# 8. Evaluate Model Performance
# ==========================================================

final_accuracy = accuracy_score(y_test, predictions)

print(f"\nFinal Model Accuracy: {final_accuracy:.4f}")

# ==========================================================
# 9. Confusion Matrix and Heatmap
# ==========================================================
confusion_grid = confusion_matrix(y_test, predictions)

sns.heatmap(
    confusion_grid,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris_dataset.target_names,
    yticklabels=iris_dataset.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nConfusion Matrix:")
print(confusion_grid)


# ==========================================================
# 10. Classification Report
# ==========================================================
# Includes Precision, Recall, F1-Score and Support

classification_results = classification_report(
    y_test,
    predictions,
    target_names=iris_dataset.target_names
)

print("\nClassification Report:")
print(classification_results)
