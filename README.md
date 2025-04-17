# 🚚 Anomaly Detection and Supervised Classification on Logistics Data

This project analyzes a logistics dataset to **detect shipment anomalies** and **predict delays** using both unsupervised (Isolation Forest) and supervised (XGBoost) approaches. It includes synthetic data generation to improve class balance and advanced feature engineering to enhance model performance.

---

## 📦 Project Highlights

- 🔍 **Anomaly Detection** using Isolation Forest
- 🤖 **Supervised Classification** using XGBoost
- ⚖️ **Synthetic Data Upsampling** to balance delay labels
- 🔧 **Custom Feature Engineering** from timestamp and user behavior
- 📊 **Evaluation Metrics**: Accuracy, Confusion Matrix, Classification Report
- 🌍 **Geospatial Visualization** of anomalous vs normal deliveries

---

## 📁 Dataset

- **Source**: `/kaggle/input/smart-logistics-supply-chain-dataset/smart_logistics_dataset.csv`
- Includes geolocation, environmental metrics, operational KPIs, timestamps, and delay labels

---

## 🧪 Key Steps

1. **Synthetic Resampling**: Upsample minority class (`Logistics_Delay = 1`) for balance
2. **Feature Engineering**:
   - Extract `Hour`, `Weekday`, `Month` from timestamp
   - Create `Efficiency_Ratio`, `User_Intensity`, `Weather_Severity`
3. **Preprocessing**:
   - Standard scaling for numeric features
   - One-hot encoding for categorical features
4. **Modeling**:
   - Train/test split for supervised classifier
   - Fit `XGBClassifier` on balanced data
   - Apply `IsolationForest` for unsupervised anomaly detection
5. **Evaluation**:
   - Compare accuracy and classification reports
   - Visualize anomaly distribution and decision scores
   - Map anomalies by GPS coordinates

---

## 📊 Output Examples

- 📈 XGBoost delay prediction report
- 📉 Isolation Forest anomaly detection performance
- 🗺️ Geolocation scatter plot of detected anomalies

---

## 📌 Use Cases

- Shipment delay prediction
- Supply chain risk analysis
- Anomaly detection in logistics operations
- Model comparison: supervised vs unsupervised

---

## 🛠 Tech Stack

- `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`: `IsolationForest`, preprocessing, metrics
- `xgboost` for supervised classification

---

## 📬 Contact

Feel free to reach out for questions, feedback, or collaborations!
