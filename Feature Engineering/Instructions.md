# Feature Engineering: Concepts with Code (Detailed Explanations)

---

## 🔧 1. Handling Missing Values
Missing data can break models or bias them.

**Techniques:**
- Mean/Median Imputation → Replace missing values with average or median (use median if outliers exist)
- Constant Value → Fill with 0, -1, or 'Unknown' when missing has meaning
- Drop Rows/Columns → Use when missing data is very small

**Why it matters:** Most ML models cannot handle NaN values directly.

```python
import numpy as np
import pandas as pd

# Sample dataframe
df = pd.DataFrame({
    'age': [25, 32, 47, np.nan],
    'salary': [50000, 60000, 80000, 120000],
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Chennai'],
    'gender': ['M', 'F', 'F', 'M'],
    'date': pd.to_datetime(['2020-01-01', '2021-06-15', '2022-03-10', '2023-08-20'])
})

# Fill missing values with mean
df['age'] = df['age'].fillna(df['age'].mean())

# Fill with median
df['age'] = df['age'].fillna(df['age'].median())

# Fill with constant
df['age'] = df['age'].fillna(0)

# Drop rows with missing values
df_drop = df.dropna()
```

---

## 🔠 2. Encoding Categorical Variables
Models cannot understand text data, so categories must be converted to numbers.

**Techniques:**
- Label Encoding → Assigns numbers to categories (may introduce false order)
- One-Hot Encoding → Creates binary columns (best for nominal data)
- Ordinal Encoding → Used when categories have order

**Why it matters:** Incorrect encoding can mislead the model.

```python
# Label Encoding (convert categories to numbers)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_label'] = le.fit_transform(df['city'])

# One-Hot Encoding (create binary columns)
df_ohe = pd.get_dummies(df, columns=['city', 'gender'])

# Ordinal Encoding (for ordered categories)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
df[['gender_encoded']] = oe.fit_transform(df[['gender']])
```

---

## 📏 3. Feature Scaling
Scaling ensures all features contribute equally.

**Techniques:**
- Standardization → Mean = 0, Std = 1
- Min-Max Scaling → Range [0,1]


**Why it matters:** Distance-based models depend on scale.

```python
# Standardization (mean=0, std=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['salary_scaled'] = scaler.fit_transform(df[['salary']])

# Normalization (0 to 1 range)
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
df['salary_norm'] = mm.fit_transform(df[['salary']])
```

---

## 🔄 4. Feature Transformation
Used to fix skewed or non-normal data distributions.

**Techniques:**
- Log Transform → Reduces skewness
- Square Root → Mild transformation
- Power Transform → Makes data more Gaussian-like

**Why it matters:** Many models assume normal distribution.

```python
# Log transformation (useful for skewed data)
df['salary_log'] = np.log(df['salary'])

# Square root transformation
df['salary_sqrt'] = np.sqrt(df['salary'])

# Power transformation
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer()
df['salary_power'] = pt.fit_transform(df[['salary']])
```

---

## 🧠 5. Feature Creation
Creating new features from existing ones.

**Why it matters:** Better features improve model performance more than complex models.

```python
# Create new feature from existing ones
df['income_per_age'] = df['salary'] / df['age']

```

---

## 🧮 6. Polynomial Features
Adds non-linear relationships between features.

**Why it matters:** Helps linear models capture complex patterns.

```python
from sklearn.preprocessing import PolynomialFeatures

# Fill missing values first
df[['age', 'salary']] = df[['age', 'salary']].fillna(df[['age', 'salary']].mean())

# Then apply polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['age', 'salary']])
```

---

## 📅 7. Date Feature Extraction
Extract useful components from date columns.

**Why it matters:** Raw dates are not useful, but components reveal patterns.

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
```

---

## 📊 8. Binning (Discretization)
Convert continuous variables into categories.

**Techniques:**
- Equal-width binning
- Equal-frequency binning

**Why it matters:** Reduces noise and simplifies models.

```python
# Equal-width binning
df['age_bin'] = pd.cut(df['age'], bins=3, labels=['young', 'middle', 'old'])

# Equal-frequency binning
df['age_qbin'] = pd.qcut(df['age'], q=3)
```

---

## 🚨 10. Handling Outliers
Outliers are extreme values that distort models.

**Technique:**
- IQR method

**Why it matters:** Improves model stability.

```python

# =========================
# OUTLIER REMOVAL USING IQR METHOD
# =========================

# Step 1: Calculate Q1 (25th percentile)
# This means 25% of the salary values are below this number
Q1 = df['salary'].quantile(0.25)

# Step 2: Calculate Q3 (75th percentile)
# This means 75% of the salary values are below this number
Q3 = df['salary'].quantile(0.75)

# Step 3: Calculate IQR (Interquartile Range)
# This represents the spread of the middle 50% of the data
IQR = Q3 - Q1

# Step 4: Define lower and upper bounds for outliers
# Any value outside this range is considered an outlier
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 5: Filter the dataset
# Keep only values within the normal range (remove outliers)
df_out = df[
    (df['salary'] >= lower_bound) & 
    (df['salary'] <= upper_bound)
]

# =========================
# SUMMARY:
# - Q1 → 25% data point
# - Q3 → 75% data point
# - IQR → middle 50% spread
# - 1.5 * IQR rule → standard way to detect outliers
# - df_out → cleaned dataset without extreme values
# =========================
```

---

## 📝 11. Text Feature Engineering
Convert text into numerical features.

**Techniques:**
- Bag of Words : Converts text into numbers by counting how many times each word appears.
- TF-IDF : Converts text into numbers by weighting words based on importance (frequent in a document but rare across documents)

**Why it matters:** Enables NLP tasks.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

text_data = ["machine learning is fun", "feature engineering is important"]

# =========================
# Bag of Words
# =========================
cv = CountVectorizer()
bow = cv.fit_transform(text_data)

# Print vocabulary (words)
print("BoW Vocabulary:", cv.get_feature_names_out())

# Print matrix (converted to array for readability)
print("BoW Matrix:\n", bow.toarray())

# =========================
# TF-IDF
# =========================
tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(text_data)

# Print vocabulary
print("\nTF-IDF Vocabulary:", tfidf.get_feature_names_out())

# Print matrix
print("TF-IDF Matrix:\n", tfidf_features.toarray())
```

---

## 📉 12. Dimensionality Reduction
Reduce number of features while keeping important information.

**Technique:** PCA

**Why it matters:** Improves speed and reduces redundancy.

```python
from sklearn.decomposition import PCA

# Define X (your feature matrix)
X = df[['age', 'salary']]   # select numeric columns

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Print result
print(X_pca)
```

---

## 🔗 13. Interaction Features
Combine features to capture relationships.

**Why it matters:** Models learn feature interactions.

```python
# Multiply features to capture interaction
df['age_salary_interaction'] = df['age'] * df['salary']
```

---

## 🎯 14. Target Encoding
Replace category with mean target value.

**Why it matters:** Useful for high-cardinality data.

```python
# =========================
# TARGET ENCODING
# =========================

# Group the data by 'city'
# For each city, calculate the average (mean) salary
# Then assign that mean value back to every row of that city

df['city_target_enc'] = df.groupby('city')['salary'].transform('mean')

# Example:
# Suppose data is:
# city     salary
# Delhi    50000
# Delhi    60000
# Mumbai   70000

# Then:
# Delhi mean = (50000 + 60000) / 2 = 55000
# Mumbai mean = 70000

# Result:
# city     salary   city_target_enc
# Delhi    50000    55000
# Delhi    60000    55000
# Mumbai   70000    70000
```

---

## 🔢 15. Frequency Encoding
Replace category with occurrence count.

**Why it matters:** Simple and effective for large datasets.

```python
# =========================
# FREQUENCY ENCODING
# =========================

# Step 1: Count how many times each category appears
# Example: Delhi → 2 times, Mumbai → 1 time
freq = df['city'].value_counts()

# Step 2: Replace each category with its frequency
# Each city name is mapped to its count
df['city_freq'] = df['city'].map(freq)

# Example:
# Original data:
# city
# Delhi
# Delhi
# Mumbai

# freq:
# Delhi → 2
# Mumbai → 1

# Result:
# city     city_freq
# Delhi    2
# Delhi    2
# Mumbai   1
```

---

## ⚙️ 16. Custom Features
Create features using domain knowledge.

**Why it matters:** Domain knowledge can significantly boost performance.

```python
# =========================
# CUSTOM FEATURE USING LAMBDA
# =========================

# Apply a function to each value in the 'age' column
# lambda x → takes each age value one by one

# Condition:
# if age < 30 → assign 'young'
# else → assign 'old'

df['age_group'] = df['age'].apply(lambda x: 'young' if x < 30 else 'old')

# Example:
# age
# 25 → 'young'
# 32 → 'old'
# 18 → 'young'
# 45 → 'old'

# Result:
# age   age_group
# 25    young
# 32    old
# 18    young
# 45    old
```

---

## 🧹 17. Dropping Useless Features
Remove irrelevant or redundant columns.

**Why it matters:** Reduces noise and improves efficiency.

```python
df = df.drop(columns=['date'])
```

---

# End of Guide
