# AnomaData-Automated_Anomaly_Detection_for_Predictive_Maintenance
This project focuses on predictive maintenance by identifying anomalies in sensor data using machine learning. It includes EDA, feature engineering, handling class imbalance, and model training with Logistic Regression, Decision Tree, and Random Forest.

### CAPSTONE PROJECT
# **AnomaData - Automated Anomaly Detection for Predictive Maintenance**

## **Overview**
AnomaData is a machine learning project designed to predict anomalies in sensor data for predictive maintenance. By identifying equipment anomalies early, businesses can reduce risks, prevent unexpected downtime, and improve overall operational efficiency. 

This project leverages statistical analysis, feature engineering, advanced visualization techniques, and machine learning models to develop a robust anomaly detection system.

---

## **Features**
- **Exploratory Data Analysis (EDA):** Visualizations to understand data trends, relationships, and outliers.
- **Preprocessing Pipeline:** Automated handling of missing values, scaling, and feature engineering.
- **Imbalanced Data Handling:** Using SMOTE for balancing the dataset to improve model performance.
- **Model Training and Tuning:**
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - Other models for experimentation.
- **Hyperparameter Tuning:** Optimize models using GridSearchCV and RandomizedSearchCV.
- **Evaluation Metrics:** Precision, recall, F1-score, ROC curves, and confusion matrices to assess model performance.
- **Time Series Analysis:** Exploration of temporal trends in anomalies.
- **Feature Importance:** Insights into which features contribute the most to anomaly detection.

---

## **Problem Statement**
Equipment failure in various industries poses significant risks and costs. Predictive maintenance aims to address this by evaluating the condition of equipment through real-time monitoring. This project focuses on developing an anomaly detection system to predict and prevent such failures.

---

## **Data**
The dataset includes:
- **Sensor readings:** Continuous variables representing real-time equipment metrics.
- **Target labels:** Binary values (`y`), where `1` indicates an anomaly and `0` indicates normal behavior.
- **Datetime:** Timestamps for time series analysis.

Key characteristics:
- ~18,000 rows of data.
- 60+ features (sensor readings, derived metrics).
- Highly imbalanced (anomalies account for <1% of the data).

---

## **Steps and Workflow**
### **1. Exploratory Data Analysis (EDA)**
- Visualize feature distributions, correlations, and target variable relationships.
- Detect and handle outliers using IQR-based methods.
- Generate insights into time-based patterns in anomalies.

### **2. Preprocessing**
- Time-based feature extraction (hour, day of the week, etc.).
- Scaling and normalization for numerical features.
- Handling class imbalance using SMOTE.

### **3. Model Training and Evaluation**
- Train multiple machine learning models to identify anomalies.
- Use hyperparameter tuning for optimal performance.
- Evaluate models using metrics like precision, recall, F1-score, and ROC-AUC.

### **4. Advanced Visualizations**
- Trends in anomalies over time.
- Heatmaps for hourly and daily patterns.
- Feature importance for tree-based models.

### **5. Model Deployment Plan**
- Package the model and preprocessing pipeline.
- Provide recommendations for deploying the solution in a production environment.

---

## **Technologies Used**
- **Python:** Core language for data analysis and modeling.
- **Libraries:**
  - `pandas`, `numpy`: Data manipulation and processing.
  - `matplotlib`, `seaborn`: Advanced data visualizations.
  - `scikit-learn`: Machine learning models and preprocessing.
  - `imblearn`: Handling imbalanced datasets.
  - `xgboost`: Advanced tree-based modeling.
- **Google Colab:** Cloud-based development environment.

---

## **Installation and Usage**
### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/AnomaData.git
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Notebook**
Open the `AnomaData.ipynb` file in Google Colab or Jupyter Notebook.

### **4. Data Input**
Place the dataset in the appropriate folder as mentioned in the code.

---

## **Project Structure**
```
AnomaData/
│
├── data/                      # Dataset folder
│   ├── AnomaData.xlsx
│   └── ...
├── visuals/                   # Visualization outputs
│   ├── histograms.png
│   ├── time_series_anomalies.png
│   └── ...
├── src/                       # Source code for the project
│   ├── preprocessing.py       # Preprocessing functions
│   ├── modeling.py            # Model training and evaluation
│   └── ...
├── README.md                  # Project overview and instructions
└── AnomaData.ipynb            # Main notebook
```

---

## **Results**
- **Accuracy:** Achieved ~90% accuracy on balanced data.
- **F1-Score for Anomalies:** Significant improvement after using SMOTE.
- **Insights:** Identified key features contributing to anomaly detection, such as `x43`, `x49`, and `x55`.

---

## **Future Enhancements**
1. **Deployment:** Package the model as an API for real-time anomaly detection.
2. **Deep Learning Models:** Experiment with LSTM or autoencoders for better time series modeling.
3. **Real-time Data Handling:** Integrate streaming data pipelines for continuous monitoring.

---

## **Contributions**
Feel free to fork this repository and submit pull requests. Contributions are welcome!

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.
