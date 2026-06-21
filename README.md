# Heart Disease Prediction MLOps Project 
## 📊 Section 5: Data Drift Analysis (Evidently)

### 1. Which features showed drift and why?
The features `age` and `chol` (cholesterol) both showed statistically significant data drift during monitoring. This occurred because our production data simulation introduced a demographic shift, representing an incoming patient group that is slightly older and exhibits higher average cholesterol levels than the original baseline training set.

### 2. Would this drift likely affect model performance?
At a 15.38% overall feature drift share, the dataset is still considered structurally stable. However, because both age and cholesterol are high-importance clinical indicators for heart disease classification, a continued upward drift in these features could eventually lead to model degradation (prediction bias or underestimation of risk) if left unchecked.

### 3. What action would you recommend?
**Continue Monitoring.** Since the overall dataset drift status is marked as stable and falls safely below our operational threshold of 30%, immediate retraining is not required. We should continue monitoring the data stream, but prepare to trigger an automated retraining pipeline if the drift share crosses the 30% threshold or if model accuracy drops.