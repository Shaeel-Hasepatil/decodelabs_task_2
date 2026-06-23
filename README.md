# 🌸 Iris Flower Classification using K-Nearest Neighbors (KNN)

This project was developed as part of the **DecodeLabs Internship - Project 2**.

The goal of this project is to classify Iris flowers into different species using the **K-Nearest Neighbors (KNN)** machine learning algorithm. The project demonstrates the complete machine learning workflow, including data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

---

## 🚀 Project Overview

The Iris dataset contains measurements of flower characteristics:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

Using these features, the model predicts one of the following species:

* Setosa
* Versicolor
* Virginica

---

## 📊 Machine Learning Workflow

1. Load and explore the Iris dataset
2. Split data into training and testing sets
3. Apply feature scaling using StandardScaler
4. Train KNN models with different K values
5. Find the optimal K based on accuracy
6. Plot K vs Accuracy (Elbow Curve)
7. Train the final model using the best K
8. Generate predictions
9. Evaluate model performance
10. Visualize results using a confusion matrix

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```text
Project-2_KNN_Classification/
│
├── main.py
├── README.md
├── requirements.txt
├── .gitignore
└── screenshots/
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone <repository-url>
cd Project-2_KNN_Classification
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py
```

---

## 📈 Model Training

The project automatically tests K values from 1 to 20 and selects the K value that produces the highest classification accuracy.

```python
for K in range(1, 21):
```

The optimal K is then used to train the final KNN classifier.

---

## 📉 Visualizations

### K vs Accuracy Plot

Displays model performance across different K values to help identify the optimal number of neighbors.
<p align="center">
  <img src="screenshots/K_vs_Accuracy_plot.png" width="45%">
</p>

### Confusion Matrix

Visualizes prediction performance across all flower species using a heatmap.
<p align="center">
  <img src="screenshots/Confusion_Matrix.png" width="45%">
</p>
---

## 📋 Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1-Score
* Support

---

## 🎯 Learning Outcomes

This project demonstrates:

* Supervised Machine Learning
* Classification Problems
* K-Nearest Neighbors Algorithm
* Data Preprocessing
* Feature Scaling
* Hyperparameter Tuning
* Model Evaluation
* Data Visualization

---

## 👨‍💻 Author

Shaeel Hasepatil

DecodeLabs Internship — Project 2
