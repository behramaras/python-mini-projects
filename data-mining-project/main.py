import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# VERİ YÜKLEME VE TEMİZLEME
# ==============================================

# Excel dosyasından veriyi yüklüyoruz
df = pd.read_excel("dataset for mendeley 181220.xlsx")

# Sütun isimlerindeki gereksiz boşlukları ve tırnak işaretlerini temizliyoruz
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Analizde kullanacağımız sütunları belirliyoruz
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

# Sadece gerekli sütunları alıp, eksik değerleri çıkarıyoruz
df = df[required_columns].dropna()

# Sınav sütunlarını belirliyoruz (Math, Science, English içeren sütunlar)
exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]

# Tüm sınavların ortalamasını hesaplıyoruz
df['ExamAverage'] = df[exam_cols].mean(axis=1)
warnings.filterwarnings('ignore')

# ==============================================
# VERİ YÜKLEME VE TEMİZLEME
# ==============================================

# Excel dosyasından veriyi yüklüyoruz
df = pd.read_excel("dataset for mendeley 181220.xlsx")

# Sütun isimlerindeki gereksiz boşlukları ve tırnak işaretlerini temizliyoruz
df.columns = df.columns.str.strip().str.replace("'", "", regex=False).str.replace('"', "", regex=False)

# Analizde kullanacağımız sütunları belirliyoruz
required_columns = [
    'Gender',
    'Age as of Academic Year 17/18',
    'Previous Curriculum (17/18)2',
    'Math20-1 ', 'Science20-1 ', 'English20-1 ',
    'Math20-2 ', 'Science20-2 ', 'English20-2 ',
    'Math20-3 ', 'Science20-3 ', 'English20-3 '
]

# Sadece gerekli sütunları alıp, eksik değerleri çıkarıyoruz
df = df[required_columns].dropna()

# Sınav sütunlarını belirliyoruz (Math, Science, English içeren sütunlar)
exam_cols = [col for col in df.columns if "Math" in col or "Science" in col or "English" in col]

# Tüm sınavların ortalamasını hesaplıyoruz
df['ExamAverage'] = df[exam_cols].mean(axis=1)

# Yerleştirme seviyesini belirleyen fonksiyon tanımlıyoruz
def get_placement_level(avg):
    """
    Sınav ortalamasına göre yerleştirme seviyesini belirler
    85+ : High (Yüksek)
    75-84: Medium (Orta)  
    75 altı: Low (Düşük)
    """
    if avg >= 85:
        return 'High'
    elif avg >= 75:
        return 'Medium'
    else:
        return 'Low'

# Her öğrenci için yerleştirme seviyesini hesaplıyoruz
df['PlacementLevel'] = df['ExamAverage'].apply(get_placement_level)
