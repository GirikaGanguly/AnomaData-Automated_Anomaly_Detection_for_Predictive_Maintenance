# **AnomaData - Automated Anomaly Detection for Predictive Maintenance**

## **Author:**
**GIRIKA GANGULY**  

---

## **Project Link**  
[GitHub Repository: AnomaData](https://github.com/GirikaGanguly/AnomaData-Automated_Anomaly_Detection_for_Predictive_Maintenance)

---

## **Introduction**

### **Problem Statement**  
Predictive maintenance is critical for minimizing downtime and preventing costly equipment failures. This project focuses on building a machine learning pipeline to identify anomalies in machine behavior using historical sensor data. The goal is to predict potential breakdowns by accurately detecting anomalies, represented by the binary target variable `y` (1 = anomaly, 0 = normal).

### **Objective**  
Develop a robust and interpretable anomaly detection model that:  
- Achieves an **F1-score** of at least **0.75** on the anomaly class.  
- Addresses the challenges posed by class imbalance.  
- Can be deployed as a predictive maintenance tool for industrial systems.  

**Dataset:** : [Anoma_data.csv](https://kh3-ls-storage.s3.us-east-1.amazonaws.com/DS%20Project%20Guide%20Data%20Set/AnomaData.xlsx)


---

## **Workflow and Methodology**

### **1. Exploratory Data Analysis (EDA)**  
- **Imbalanced Dataset:** Anomalies account for <1% of the data.  
- **Skewness:** Many numerical features exhibit high skewness, requiring transformation.  
- **Outliers:** Detected and handled to minimize their effect.  
- **Class Distribution:**
  - Normal: ~18,000 samples.  
  - Anomaly: ~120 samples.  

**Key Visualizations:**  
- Heatmaps for feature correlations.  
- Pair plots and scatter plots for separability analysis.  
- Time series plots for detecting anomaly patterns.  

---

### **2. Preprocessing Pipeline**  
1. **Datetime Features:** Removed irrelevant datetime columns.  
2. **Duplicates:** Identified and removed duplicate rows.  
3. **Missing Values:** Imputed missing values using mean imputation.  
4. **Skewness Correction:** Applied Yeo-Johnson transformation for skewed features.  
5. **Outlier Handling:** IQR-based capping to limit extreme values.  
6. **Scaling:** Standardized numerical features for model consistency.  
7. **Class Imbalance:**  
   - Used **SMOTE** for oversampling anomalies.  
   - Applied `scale_pos_weight` in **XGBoost** to address imbalance.  

---

### **3. Modeling**  

#### **Baseline Models:**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

#### **Ensemble Learning:**  
- Combined **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** in a Voting Classifier.  
- Weighted soft voting was used to leverage XGBoost's strengths.  

**Hyperparameter Tuning:**  
- Conducted **GridSearchCV** for model optimization.  
- Tuned Voting Classifier weights for best performance.  

---

### **4. Evaluation**  

**Metrics Used:**  
- Accuracy, Precision, Recall, F1-Score, AUC-ROC.  

**Key Visualizations:**  
- **Confusion Matrices**: Analyze false positives/negatives.  
- **ROC Curves**: Assess model discrimination.  
- **Learning Curves**: Compare training and validation performance.  

---

## **Results**

### **Model Performance**  

| **Model**                 | **Precision (1)** | **Recall (1)** | **F1-Score (1)** | **AUC-ROC** |  
|---------------------------|------------------:|--------------:|----------------:|------------:|  
| Logistic Regression       | 0.04             | 0.8           | 0.08            | 0.93        |  
| Decision Tree             | 0.27             | 0.64          | 0.38            | 0.81        |  
| Random Forest             | 0.75             | 0.6           | 0.67            | 0.98        |  
| **XGBoost (Weighted)**    | 0.58             | 0.84          | 0.68            | 0.99        |  
| **Voting Classifier**     | **0.64**         | **0.84**      | **0.72**        | **0.97**    |  

### **Key Insights**  
- **Voting Classifier** delivered the best performance with an F1-score of **0.72** for anomalies.  
- The model achieved a **balanced precision (64%)** and recall (84%) with a robust **AUC-ROC of 0.97**.  

---

## **Conclusion**

### **Best Model:**  
The **Voting Classifier** emerged as the best-performing model, achieving a balance between precision and recall for anomalies.

### **Challenges and Solutions:**  
1. **Class Imbalance:** Addressed using **SMOTE** and weighted training.  
2. **Overfitting:** Managed through hyperparameter tuning and ensemble methods.  
3. **Precision-Recall Trade-off:** Balanced through model ensembling and weight tuning.  

---

## **Future Work**  
1. **Advanced Feature Engineering:** Create derived features and explore feature interactions.  
2. **Deep Learning Models:** Test **Autoencoders** and **LSTM** for sequential anomaly detection.  
3. **Real-time Deployment:** Package the model into an API for real-time anomaly monitoring.  
4. **Explainability:** Use SHAP or LIME for better interpretability.  
5. **Threshold Optimization:** Dynamically adjust thresholds to minimize false positives.  

---

## **How to Run the Project**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/GirikaGanguly/AnomaData-Automated_Anomaly_Detection_for_Predictive_Maintenance.git
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the Jupyter Notebook for EDA and Modeling.  

---

## **Technologies Used**  
- **Python Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SMOTE, Matplotlib, Seaborn.  
- **Tools:** Jupyter Notebook, GridSearchCV, Visualization Libraries.  

---

## **Acknowledgments**  
Special thanks to KH mentors for their continuous support and guidance throughout the capstone project.

---

## **Contributions**  
Contributions are welcome! Fork the repository, create a branch, and submit a pull request.  

---

## **Contact**  
For any queries, feel free to connect:  
- **GitHub:** [GirikaGanguly](https://github.com/GirikaGanguly)  

---

**This project showcases the potential of machine learning for predictive maintenance and anomaly detection in industrial systems.**  
