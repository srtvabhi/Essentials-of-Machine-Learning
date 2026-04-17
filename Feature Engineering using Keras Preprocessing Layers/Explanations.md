# 📘 Notebook: Keras Preprocessing Layers (PetFinder Dataset)

## 🔹 Cell [1]: Install Dependencies
```python
!pip install scikit-learn
!sudo apt-get install graphviz -y
```
- Installs required libraries:
  - scikit-learn → used for train_test_split
  - graphviz → used to visualize model architecture
- Ensures environment has all dependencies before running notebook

---

## 🔹 Cell [2]: Import Libraries
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
```
- Imports:
  - numpy → numerical operations
  - pandas → data handling
  - tensorflow → deep learning framework
  - train_test_split → dataset splitting
  - keras layers → building model
  - preprocessing → feature engineering layers

---

## 🔹 Cell [3]: Check TensorFlow Version
```python
tf.__version__
```
- Prints TensorFlow version (2.11.0)
- Ensures compatibility with preprocessing APIs

---

## 🔹 Cell [4]: Load Dataset
```python
dataset_url = 'http://storage.googleapis.com/...'
csv_file = 'gs://cloud-training/...'

tf.keras.utils.get_file(...)
dataframe = pd.read_csv(csv_file)
```
- Downloads dataset from URL
- Reads CSV into pandas DataFrame
- Data is now structured (rows & columns)

---

## 🔹 Cell [5]: Preview Data
```python
dataframe.head()
```
- Displays first 5 rows
- Helps understand structure and features

---

## 🔹 Cell [6]: Create Target Variable
```python
dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
```
- Converts problem into binary classification:
  - 0 → not adopted
  - 1 → adopted
- Drops unused columns:
  - AdoptionSpeed
  - Description

---

## 🔹 Cell [7]: Split Dataset
```python
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
```
- Splits dataset:
  - Train → 64%
  - Validation → 16%
  - Test → 20%
- Enables proper evaluation

---

## 🔹 Cell [8]: Create Input Pipeline
```python
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
```
- Converts DataFrame → tf.data.Dataset
- Steps:
  - Copy dataframe
  - Separate labels
  - Convert to tensor slices
  - Shuffle data
  - Batch data
  - Prefetch for efficiency
- Output: (features_dict, labels)

---

## 🔹 Cell [9]: Create Dataset
```python
train_ds = df_to_dataset(train, batch_size=5)
```
- Creates dataset with batch size = 5
- Small batch for demonstration

---

## 🔹 Cell [10]: Inspect Dataset
```python
[(train_features, label_batch)] = train_ds.take(1)
```
- Extracts one batch
- Shows:
  - feature keys
  - values
  - labels
- Data is dictionary format

---

## 🔹 Cell [11]: Normalization Function
```python
def get_normalization_layer(name, dataset):
```
- Creates normalization layer
- Steps:
  - Extract feature
  - Learn mean & std using adapt()
- Used for numeric features

---

## 🔹 Cell [12]: Apply Normalization
```python
layer(photo_count_col)
```
- Converts values to standardized form
- Mean ≈ 0, std ≈ 1

---

## 🔹 Cell [13]: Category Encoding Function
```python
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
```
- Converts categorical data:
  - String → integer indices
  - Integer → encoded indices
- Applies one-hot encoding

---

## 🔹 Cell [14]: Encode String Feature
```python
layer(type_col)
```
- Converts strings (Dog/Cat) → one-hot vectors

---

## 🔹 Cell [15]: Encode Integer Feature
```python
category_encoding_layer(type_col)
```
- Converts integer features into categorical encoding

---

## 🔹 Cell [16]: Recreate Dataset
```python
batch_size = 256
train_ds = df_to_dataset(...)
```
- Uses larger batch size for faster training

---

## 🔹 Cell [17]: Process Numeric Features
```python
for header in ['PhotoAmt', 'Fee']:
```
- Creates input layer
- Applies normalization
- Stores encoded features

---

## 🔹 Cell [18]: Process Integer Feature
```python
age_col = tf.keras.Input(...)
```
- Encodes Age feature

---

## 🔹 Cell [19]: Process String Features
```python
for header in categorical_cols:
```
- Encodes all categorical features
- Uses lookup + one-hot encoding

---

## 🔹 Cell [20]: Build Model
```python
all_features = concatenate(...)
Dense(32) → Dropout → Output
```
- Combines all features
- Neural network:
  - Dense layer
  - Dropout
  - Output layer

---

## 🔹 Compile Model
```python
model.compile(...)
```
- Optimizer: Adam
- Loss: BinaryCrossentropy
- Metric: Accuracy

---

## 🔹 Cell [21]: Visualize Model
```python
plot_model(model)
```
- Shows architecture graph

---

## 🔹 Cell [22]: Train Model
```python
model.fit(...)
```
- Trains model for 10 epochs
- Accuracy improves over time

---

## 🔹 Cell [23]: Evaluate Model
```python
model.evaluate(test_ds)
```
- Tests model
- Accuracy ≈ 71.8%

---

## 🔹 Cell [24]: Save & Reload Model
```python
model.save(...)
load_model(...)
```
- Saves full pipeline model
- Reloads it

---

## 🔹 Cell [25]: Make Prediction
```python
sample = {...}
model.predict(...)
```
- Predicts adoption probability
- Example: ~80% adoption chance

---

## 🧠 Final Summary

### Pipeline:
1. Load data  
2. Preprocess  
3. Create tf.data pipeline  
4. Encode features  
5. Train model  
6. Evaluate  
7. Predict  

### Key Insight:
- Preprocessing is embedded inside model
- Model can directly take raw input
- Easier deployment
