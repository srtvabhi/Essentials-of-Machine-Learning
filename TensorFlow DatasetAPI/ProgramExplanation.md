

# 📘 TensorFlow Dataset API — Full Cell-by-Cell Deep Explanation

---

## 🔹 Cell 1: Imports

### 💻 Code
```python
import json
import math
import os
from pprint import pprint
```

### 🔍 Explanation
- `json` → Used to read/write structured data (like configs or outputs)
- `math` → Provides mathematical operations (sqrt, log, etc.)
- `os` → Helps interact with file system (paths, folders)
- `pprint` → Prints complex data structures in readable format

👉 These are helper libraries (not core ML logic).

---

## 🔹 Cell 5: Creating Synthetic Data

### 💻 Code
```python
N_POINTS = 10
X = tf.constant(range(N_POINTS), dtype=tf.float32)
Y = 2 * X + 10
```

### 🔍 Step-by-Step Explanation

#### Line 1
```python
N_POINTS = 10
```
- Defines number of data samples

---

#### Line 2
```python
X = tf.constant(range(N_POINTS), dtype=tf.float32)
```

### Functions Used:
- `range(N_POINTS)`
  - Generates values: [0,1,2,...,9]

- `tf.constant()`
  - Converts Python list → TensorFlow tensor
  - Stored efficiently (CPU/GPU)

- `dtype=tf.float32`
  - Sets numeric type (important for ML models)

👉 Output:
```
X = [0.,1.,2.,3.,...,9.]
```

---

#### Line 3
```python
Y = 2 * X + 10
```

### Functions Used:
- TensorFlow broadcasting (vectorized ops)

### What happens:
- Each element processed:
```
Y[i] = 2 * X[i] + 10
```

👉 Output:
```
Y = [10,12,14,...,28]
```

---

## 🔹 Cell 8: Dataset Function

### 💻 Code
```python
def create_dataset(X, Y, epochs, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.repeat(epochs).batch(batch_size, drop_remainder=True)
    return dataset
```

---

## 🔍 FULL FUNCTION BREAKDOWN

---

### Function Definition
```python
def create_dataset(X, Y, epochs, batch_size):
```

- `X` → features
- `Y` → labels
- `epochs` → number of repetitions
- `batch_size` → size of each batch

---

### Line 1
```python
dataset = tf.data.Dataset.from_tensor_slices((X, Y))
```

### Function: `from_tensor_slices`

#### What it does:
- Splits tensors into rows
- Combines them as pairs

👉 Internally:
```
(X,Y) → (x1,y1), (x2,y2), ...
```

👉 Equivalent:
```python
zip(X, Y)
```

---

### Line 2
```python
dataset = dataset.repeat(epochs)
```

### Function: `repeat()`

#### What it does:
- Repeats dataset multiple times

Example:
```
epochs = 2
→ dataset runs twice
```

#### Important:
- No memory duplication
- Just loops iterator

---

### Line 3
```python
dataset = dataset.batch(batch_size, drop_remainder=True)
```

### Function: `batch()`

#### What it does:
- Groups elements into batches

Example:
```
batch_size = 3
→ [(x1,y1),(x2,y2),(x3,y3)]
```

---

### Argument: `drop_remainder=True`

#### Why needed:
- Last batch may be smaller

Example:
```
[1,2,3,4] with batch=3
→ [1,2,3] and [4]
```

👉 Drops `[4]`

#### Benefit:
- Ensures fixed-size batches
- Required for neural networks

---

### Final Output
```python
return dataset
```

Returns:
```
Dataset object (iterator-like)
```

---

## 🔹 Cell 9: Iteration

### 💻 Code
```python
for batch in dataset:
    print(batch)
```

---

## 🔍 What happens internally

Each batch contains:
```python
(
  tensor([x1,x2,x3]),
  tensor([y1,y2,y3])
)
```

👉 Structure:
```
(features, labels)
```

---

## 🚖 Taxi Dataset Equivalent

---

### 💻 Code
```python
dataset = tf.data.experimental.make_csv_dataset(
    "taxi-train.csv",
    batch_size=32,
    label_name="fare_amount"
)
```

---

## 🔍 Function Breakdown

---

### Function: `make_csv_dataset`

#### What it does:
- Reads CSV file
- Converts into TensorFlow dataset

---

### Arguments

#### 1. `"taxi-train.csv"`
- Input file path

---

#### 2. `batch_size=32`
- Number of rows per batch

---

#### 3. `label_name="fare_amount"`
- Separates target column

---

## 🔍 Internal Working

Each row:
```
pickup, dropoff, distance → features
fare → label
```

Converted into:
```python
(features_dict, label_tensor)
```

---

## 🧠 KEY CONCEPTS

---

### 1. Lazy Execution
- Data loads only when loop runs

---

### 2. Pipeline Model
```
Data → Slice → Repeat → Batch → Model
```

---

### 3. Why tf.data?
- Handles large datasets
- Efficient
- Supports streaming

---

## 🚀 FINAL FLOW

```
CSV → tf.data → batching → training → prediction
```
