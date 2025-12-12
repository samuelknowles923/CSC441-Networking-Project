# CSC441-Networking-Project
**Names:** Samuel Knowles & Matther Uttecht
**Course:** CSC 441 Networking and Data Communications

## Summary
This final project uses machine learning algorithms to detect anomolies in network traffic. It ha sbeen trained and tested on the KDD Cup '99 dataset. It compares the Decision Tree, Random Forest, XGBoost, KNN, Logistic Regression, and SVM algorithms to identify which is the best at detect these suspicious network patterns and has an emphasis on minimizing false positives while still trying to maintain a high accuracy.


## Dataset
Data files are sourced from Kaggle posted by user "Anush" anushonkar. 
https://www.kaggle.com/datasets/anushonkar/network-anamoly-detection?resource=download&select=Network+Anamoly+Detection.docx


## Model Evaluation
1. **Accuracy** = (TP + TN) / Total
   - Overall correctness of predictions
   
2. **Precision** = TP / (TP + FP)
   - Of predicted attacks, how many were real?
   - Lower false alarm rate
   
3. **Recall** = TP / (TP + FN)
   - Of actual attacks, how many were detected?
   - Lower miss rate

4. **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean of precision and recall
   
5. **ROC-AUC** = Area Under ROC Curve
   - Plots TPR vs. FPR at different thresholds
   - 1.0 = Perfect, 0.5 = Random, 0.0 = Worst

6. **False Positive Rate (FPR)** = FP / (FP + TN)
   - Critical for not disrupting legitimate traffic
   
7. **False Negative Rate (FNR)** = FN / (FN + TP)
   - Critical for not missing attacks


## Installation and Usage
**1. Clone Repository**
```bash
git clone https://github.com/samuelknowles923/CSC441-Networking-Project.git
cd CSC441-Networking-Project
```

**2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Pipeline**
```bash
python main.py
```

**5. View Results**
- Model data is stored in each respective file in the /models folder
- Post-processing data is found in .csv files in /processed_data folder
- Images for review are found in /images folder
