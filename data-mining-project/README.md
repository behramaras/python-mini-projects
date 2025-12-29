# Student Placement Level Prediction using Random Forest

This project aims to **predict student placement levels (Low, Medium, High)** based on demographic information and exam scores using a **Random Forest Classifier**.  
The workflow includes **data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and performance evaluation**.

---

## Dataset

- **File name:** `dataset.xlsx`
- **Format:** Excel
- **Key features used:**
  - Gender  
  - Age as of Academic Year 17/18  
  - Previous Curriculum (17/18)  
  - Math, Science, and English exam scores (multiple levels)

### Target Variable
- **PlacementLevel**
  - `High` (ExamAverage ≥ 85)
  - `Medium` (75 ≤ ExamAverage < 85)
  - `Low` (ExamAverage < 75)

---

## Project Workflow

### 1. Data Loading & Cleaning
- Load data from Excel
- Clean column names
- Select required columns
- Remove missing values
- Compute overall exam average

### 2. Feature Engineering
- Calculate `ExamAverage`
- Generate `PlacementLevel` based on predefined thresholds
- Apply one-hot encoding to categorical variables

### 3. Exploratory Data Analysis (EDA)
- Placement level distribution
- Exam average distribution
- Comparisons by gender and curriculum
- Correlation matrix of numerical features

### 4. Model Training
- Algorithm: **Random Forest Classifier**
- Data split:
  - 70% Training
  - 10% Validation
  - 20% Test
- Hyperparameter tuning using **GridSearchCV**

### 5. Model Evaluation
The following metrics are calculated on the test set:
- Accuracy  
- Precision (macro)  
- Recall / Sensitivity (macro)  
- F1-Score (macro)  
- AUC (multi-class, OVR)  
- Specificity (macro)

### 6. Visualization
- Confusion Matrix
- ROC Curves (Train vs Test)
- Performance Metrics Bar Chart
- Feature Importance Ranking

---

## Output Visualizations

- Placement level class distribution
- Exam score histograms
- Boxplots by gender and curriculum
- Correlation heatmap
- Confusion matrix
- ROC curves for each class
- Feature importance bar chart

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## How to Run

1. Place `dataset.xlsx` in the project directory
2. Install required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3. Run the Python script:
    ```bash
    python main.py
     ```

