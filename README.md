# 🚬 Smoking Risk Prediction using Logistic Regression

## 📘 Overview
This project aims to predict whether a person is **at risk of developing cancer due to smoking** habits.  
It uses a **Logistic Regression model** trained on synthetic or real-world health and lifestyle data.  
The project is implemented in a Jupyter Notebook: `smoking_risk.ipynb`.

---

## 🎯 Objective
To build and evaluate a machine learning model that classifies individuals as **"High Risk"** or **"Low Risk"** for cancer, based on smoking habits and related features.

---

## 🧩 Dataset Description
The dataset used in this project (`smoking_cancer_risk.csv`) includes the following columns:

| Feature | Description |
|----------|--------------|
| `Age` | Age of the person |
| `Gender` | Encoded as Male = 1, Female = 0 |
| `Cigarettes_per_day` | Number of cigarettes smoked daily |
| `Years_smoked` | Number of years the person has smoked |
| `Alcohol_consumption` | Average daily alcohol consumption (in ml) |
| `Exercise_level` | Hours of exercise per week |
| `BMI` | Body Mass Index |
| `Cancer_risk` | Target variable (1 = High Risk, 0 = Low Risk) |

---

## 🧠 Model Used
### Logistic Regression
A classification algorithm that predicts the probability of a binary outcome (here: cancer risk) based on input features.

**Why Logistic Regression?**
- It’s simple yet effective for binary classification.
- Works well when the relationship between variables and the target is approximately linear in the log-odds space.

---

## ⚙️ Installation & Requirements
Install the required libraries before running the notebook:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 🧹 Data Preprocessing
Key preprocessing steps in the notebook include:
1. Loading the dataset using `pandas.read_csv()`.
2. Handling missing or inconsistent data (if any).
3. Encoding gender:
   ```python
   df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
   ```
4. Splitting data into training and test sets using `train_test_split`.

---

## 🧠 Model Training & Evaluation
### Model Training
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```
### Model Evaluation
```python
from sklearn.metrics import accuracy_score, precision_score,  confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

---

## 📊 Visualization
Visual analysis is performed using **Matplotlib** and **Seaborn**, including:
- Confusion Matrix heatmap
- Feature correlation heatmap
- Distribution plots for smoking habits

Example:
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="coolwarm")
plt.title("Confusion Matrix")
plt.show()
```

---

## 🔮 Predicting for a New Person
You can test the model with new input data:
```python
import pandas as pd



prediction = model.predict(new_person)
print("Predicted Risk:", "High Risk" if prediction[0] == 1 else "Low Risk")
```

---

## 🧾 Key Metrics Explained
| Metric | Description |
|---------|--------------|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Fraction of true positive predictions among all positive predictions |

---

## 📁 Project Structure
```
Smoking-Risk-Prediction/
│
├── smoking_risk.ipynb
├── data/
│   └── smoking_cancer_risk.csv
├── README.md
└── requirements.txt
```

---

## 💡 Future Scope
- Experiment with **Random Forest**, **SVM**, or **XGBoost Classifier**.
- Add medical/lifestyle factors (diet, stress levels, etc.).
- Deploy the model as a **Flask or Streamlit web app**.

---

## 👨‍💻 Author
**Mukul Kumar**  
B.Tech (CSE), NIT Sikkim  
📧 Email: *nitsikkim.mukul@example.com*  
Project: *Smoking & Cancer Risk Prediction using Logistic Regression*
