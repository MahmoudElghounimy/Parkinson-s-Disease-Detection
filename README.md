### **Parkinson’s Disease Detection Using SVM**  
 

---

## **Dataset Information**  
- The dataset used is **parkinsons.data**, which contains **biomedical voice measurements** collected from individuals.  
- The target variable is **"status"** (0 = Healthy, 1 = Parkinson’s).  

---


### **1. Data Loading and Preprocessing**  
- Reads the dataset and checks for missing values.  
- Removes **irrelevant columns ("name")**.  
- Splits the data into **features (X) and labels (Y)**.  
- Standardizes the features using **StandardScaler** for better model performance.  

### **2. Train-Test Split**  
- Splits data into **80% training** and **20% testing** sets.  

### **3. Train the SVM Model**  
- Uses a **linear kernel SVM** for classification.  
- Fits the model on the **training dataset**.  

### **4. Evaluate Model Performance**  
- Predicts accuracy on both **training and test data**.  

### **5. Make a Prediction for a New Patient**  
- Takes a **new input sample** of voice measurements.  
- Standardizes the input and predicts whether the person has **Parkinson’s Disease** or not.  

---

## **Code Breakdown**  

### **1. Import Required Libraries**  
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
```
- `numpy` & `pandas`: Data handling.  
- `train_test_split`: Splits data into training/testing sets.  
- `StandardScaler`: Standardizes feature values.  
- `svm.SVC`: Support Vector Classifier.  
- `accuracy_score`: Evaluates model performance.  

---

### **2. Load and Explore the Dataset**  
```python
parkinsons_data = pd.read_csv('parkinsons.data')

parkinsons_data.head()
parkinsons_data.shape
parkinsons_data.info()
parkinsons_data.isnull().sum()
parkinsons_data.describe()
parkinsons_data['status'].value_counts()
parkinsons_data.groupby('status').mean(numeric_only=True)
```
- Reads the dataset.  
- Checks **missing values, dataset size, and summary statistics**.  
- Analyzes **Parkinson’s distribution** (0 = Healthy, 1 = Parkinson’s).  

---

### **3. Prepare Data for Training**  
```python
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
Y = parkinsons_data['status']
```
- Drops the **"name"** column (not useful for prediction).  
- **X** contains features, **Y** contains the target label.  

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
```
- Splits data into **80% training** and **20% testing**.  

```python
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
- **Standardizes** feature values for better model performance.  

---

### **4. Train the SVM Model**  
```python
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)
```
- Uses **SVM with a linear kernel** to classify patients.  

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data = ', training_data_accuracy)
```
- Computes **accuracy on training data**.  

```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data = ', test_data_accuracy)
```
- Evaluates model performance on **test data**.  

---

### **5. Predict Parkinson’s for a New Patient**  
```python
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)
```
- A **sample patient's biomedical voice data**.  

```python
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = model.predict(std_data)
print(prediction)
```
- Converts input into **numpy array**.  
- **Reshapes** data for model input.  
- **Standardizes** input for consistency.  
- **Predicts** whether the person has Parkinson’s or not.  

```python
if (prediction[0] == 0):
  print("The Person does not have Parkinson’s Disease")
else:
  print("The Person has Parkinson’s")
```
- Outputs the **final prediction**.  

