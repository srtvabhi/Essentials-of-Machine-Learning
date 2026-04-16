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
# SelectKBest (statistical test)
from sklearn.feature_selection import SelectKBest, f_regression

X = df[['age', 'salary']]
y = df['salary']

selector = SelectKBest(score_func=f_regression, k=1)
X_new = selector.fit_transform(X, y)

# Remove outliers using IQR
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

df_out = df[(df['salary'] >= Q1 - 1.5 * IQR) & (df['salary'] <= Q3 + 1.5 * IQR)]
```

---

## 📝 11. Text Feature Engineering
Convert text into numerical features.

**Techniques:**
- Bag of Words
- TF-IDF

**Why it matters:** Enables NLP tasks.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

text_data = ["machine learning is fun", "feature engineering is important"]

# Bag of Words
cv = CountVectorizer()
bow = cv.fit_transform(text_data)

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_features = tfidf.fit_transform(text_data)
```

---

## 📉 12. Dimensionality Reduction
Reduce number of features while keeping important information.

**Technique:** PCA

**Why it matters:** Improves speed and reduces redundancy.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
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
# Replace category with mean target value
df['city_target_enc'] = df.groupby('city')['salary'].transform('mean')
```

---

## 🔢 15. Frequency Encoding
Replace category with occurrence count.

**Why it matters:** Simple and effective for large datasets.

```python
# Replace category with frequency count
freq = df['city'].value_counts()
df['city_freq'] = df['city'].map(freq)
```

---

## ⚙️ 16. Custom Features
Create features using domain knowledge.

**Why it matters:** Domain knowledge can significantly boost performance.

```python
# Create custom transformation using lambda
df['age_group'] = df['age'].apply(lambda x: 'young' if x < 30 else 'old')
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
