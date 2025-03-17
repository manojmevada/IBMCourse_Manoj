# Feature Scaling Techniques in Data Wrangling using Python

## Introduction
Feature scaling is a crucial step in data preprocessing, ensuring that numerical values across different features are on a comparable scale. This prevents certain variables from dominating others in machine learning models. In this article, we will explore the most common feature scaling techniques, their use cases, and how to implement them in Python using **NumPy** and **scikit-learn**.

---

## 1. Min-Max Scaling (Normalization)

### **Formula:**
\[
X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}
\]

- **Best used when** data has known bounds (e.g., pixel values in images).
- **Transforms values to a fixed range** (typically [0,1] or [-1,1]).

### **Implementation using NumPy**:
```python
import numpy as np

data = np.array([50, 20, 30, 80, 100])
min_val = np.min(data)
max_val = np.max(data)
normalized_data = (data - min_val) / (max_val - min_val)
print(normalized_data)
```

### **Implementation using Scikit-Learn**:
```python
from sklearn.preprocessing import MinMaxScaler

data = np.array([50, 20, 30, 80, 100]).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data)
print(normalized_data.flatten())
```

---

## 2. Z-Score Standardization (StandardScaler)

### **Formula:**
\[
X_{scaled} = \frac{X - \mu}{\sigma}
\]
- **Best used when** data follows a normal distribution.
- **Centers data around 0 with unit variance**.

### **Implementation using NumPy**:
```python
data = np.array([50, 20, 30, 80, 100])
mean = np.mean(data)
std_dev = np.std(data)
standardized_data = (data - mean) / std_dev
print(standardized_data)
```

### **Implementation using Scikit-Learn**:
```python
from sklearn.preprocessing import StandardScaler

data = np.array([50, 20, 30, 80, 100]).reshape(-1, 1)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data)
print(standardized_data.flatten())
```

---

## 3. Robust Scaling (Median & IQR)

### **Formula:**
\[
X_{scaled} = \frac{X - \text{median}}{\text{IQR}}
\]
- **Best used when** data contains extreme outliers.
- **Uses median and interquartile range (IQR) for scaling.**

### **Implementation using NumPy**:
```python
data = np.array([50, 20, 30, 80, 100, 1000])  # Data with an outlier
median = np.median(data)
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
robust_scaled_data = (data - median) / iqr
print(robust_scaled_data)
```

### **Implementation using Scikit-Learn**:
```python
from sklearn.preprocessing import RobustScaler

data = np.array([50, 20, 30, 80, 100, 1000]).reshape(-1, 1)
scaler = RobustScaler()
robust_scaled_data = scaler.fit_transform(data)
print(robust_scaled_data.flatten())
```

---

## 4. Max Abs Scaling

### **Formula:**
\[
X_{scaled} = \frac{X}{X_{max}}
\]
- **Best used when** data is centered around zero but has varying magnitudes.

### **Implementation using NumPy**:
```python
data = np.array([-500, -300, 100, 500, 700])
max_abs = np.max(np.abs(data))
max_abs_scaled_data = data / max_abs
print(max_abs_scaled_data)
```

### **Implementation using Scikit-Learn**:
```python
from sklearn.preprocessing import MaxAbsScaler

data = np.array([-500, -300, 100, 500, 700]).reshape(-1, 1)
scaler = MaxAbsScaler()
max_abs_scaled_data = scaler.fit_transform(data)
print(max_abs_scaled_data.flatten())
```

---

## 5. Log Transformation

### **Formula:**
\[
X_{scaled} = \log(X + 1)
\]
- **Best used when** data is highly skewed (exponential distribution).

### **Implementation using NumPy**:
```python
data = np.array([1, 10, 100, 1000, 10000])
log_transformed_data = np.log1p(data)  # log(1 + X) prevents log(0) error
print(log_transformed_data)
```

### **Implementation using Pandas**:
```python
import pandas as pd

df = pd.DataFrame({'Values': [1, 10, 100, 1000, 10000]})
df['Log_Transformed'] = np.log1p(df['Values'])
print(df)
```

---

## NumPy & Pandas Cheatsheet for Data Wrangling

### **NumPy**
| Function | Description |
|----------|-------------|
| `np.mean(arr)` | Calculates the mean of an array |
| `np.median(arr)` | Computes the median value |
| `np.std(arr)` | Standard deviation of the array |
| `np.percentile(arr, q)` | Computes the q-th percentile |
| `np.min(arr)` | Returns the minimum value |
| `np.max(arr)` | Returns the maximum value |
| `np.log1p(arr)` | Log transformation (log(1+x)) |

### **Pandas**
| Function | Description |
|----------|-------------|
| `df.describe()` | Summarizes numerical columns |
| `df.info()` | Displays DataFrame info |
| `df.isnull().sum()` | Checks missing values |
| `df.dropna()` | Drops missing values |
| `df.fillna(value)` | Fills missing values |
| `df.apply(func)` | Applies function to DataFrame |
| `df.groupby(column)` | Groups data by column |

---

## Choosing the Right Scaling Method

| **Method**          | **Best Use Case** |
|--------------------|------------------|
| **Min-Max Scaling**  | Data with known bounds (e.g., images). |
| **Z-Score Standardization** | Normally distributed data. |
| **Robust Scaling**   | Data with extreme outliers. |
| **Max Abs Scaling**  | Data centered around zero. |
| **Log Transformation** | Data with right-skewed distribution. |

---

## Conclusion
By applying these scaling techniques and using the key NumPy and Pandas functions, you can effectively preprocess data for machine learning models. 

Would you like real-world datasets for practice? Let me know! 🚀

