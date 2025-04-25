# 🚣 Titanic Dataset - Exploratory Data Analysis (EDA)

This repository performs exploratory data analysis on the Titanic dataset to identify key patterns, trends, and anomalies that influence survival.

---

## 📊 Summary of Identified Patterns & Trends

### 🡭 Gender (Sex)
- **Females had a much higher survival rate** than males.
  - ~75% of **females** survived.
  - Only ~18% of **males** survived.
- 🔹 _“Women and children first” policy is clearly reflected._

---

### 💼 Passenger Class (Pclass)
- **1st Class** passengers had the highest survival (~63%).
- **3rd Class** passengers had the lowest survival (~24%).
- 🔹 _Socioeconomic status influenced survival odds._

---

### 📈 Fare
- **Higher Fare → Higher survival**.
- Right-skewed distribution.
- 🔹 _Fare correlates with Pclass and indirectly with survival._

---

### 👶 Age
- **Children (<10)** had relatively higher survival.
- Most passengers were aged **20–40**.
- 🔹 _Younger age improved survival; many missing age values._

---

### ⚓ Embarked
- **Cherbourg (C)** port had higher survival than Southampton (S).
- 🔹 _Embarkation port is linked to class distribution._

---

### 👨‍👧‍👧 Family Size (SibSp + Parch)
- Passengers with **1–2 family members** had better survival.
- Solo travelers and large families had lower survival rates.
- 🔹 _Small family groups had a survival advantage._

---

## ⚠️ Anomalies & Issues
- `Fare`: extreme outliers (> $500)
- `Age`: missing data (~20%)
- `Cabin`: over 75% missing
- `Ticket`: inconsistent format, mostly unhelpful

---

## 🐍 Python Code Snippets

### Load Dataset
```python
import pandas as pd
df = pd.read_csv('titanic.csv')
```

### Barplot: Survival Rate by Gender
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Sex")
plt.show()
```

### Barplot: Survival Rate by Class
```python
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()
```

### Distribution: Fare
```python
sns.histplot(df['Fare'], kde=True, bins=30)
plt.title("Fare Distribution")
plt.show()
```

### Boxplot: Age vs Survival
```python
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()
```

### Correlation Heatmap
```python
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

---

## 📌 Notes
- Dataset used: [Titanic - Kaggle](https://www.kaggle.com/c/titanic/data)
- Libraries: `pandas`, `seaborn`, `matplotlib`

---

## 💬 Summary
EDA reveals key survival factors:
- Gender (female)
- Passenger class (1st)
- Younger age
- Smaller family
- Higher fare

These insights help in feature selection and preprocessing for ML models.

---

## 📌 License
MIT

