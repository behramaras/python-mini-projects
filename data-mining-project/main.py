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

# ==============================================
# VERİ MADENCİLİĞİ GÖRSELLEŞTİRME (EDA)
# ==============================================

# Yerleştirme seviyelerinin dağılımını görselleştiriyoruz
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='PlacementLevel', order=['Low', 'Medium', 'High'], palette='Set2')
plt.title("PlacementLevel Sınıf Dağılımı")
plt.tight_layout()
plt.show()

# Sınav ortalamalarının histogramını çiziyoruz
plt.figure(figsize=(6, 4))
sns.histplot(df['ExamAverage'], kde=True, bins=20, color='skyblue')
plt.title("ExamAverage Dağılımı")
plt.tight_layout()
plt.show()

# Cinsiyete göre sınav ortalamalarını karşılaştırıyoruz
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x='Gender', y='ExamAverage', palette='pastel')
plt.title("Cinsiyete Göre Ortalama Notlar")
plt.tight_layout()
plt.show()

# Önceki müfredata göre sınav ortalamalarını karşılaştırıyoruz
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='Previous Curriculum (17/18)2', y='ExamAverage', palette='muted')
plt.title("Müfredata Göre Ortalama Notlar")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sayısal değişkenler arasındaki korelasyonu inceliyoruz
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title("Sayısal Özellikler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# ==============================================
# MODELLEME – RANDOM FOREST
# ==============================================

# Bağımsız değişkenler (X) ve hedef değişken (y) ayırıyoruz
X = df.drop(['ExamAverage', 'PlacementLevel'], axis=1)  # ExamAverage ve PlacementLevel'ı çıkarıyoruz
y = df['PlacementLevel']  # Tahmin etmek istediğimiz değişken

# Kategorik değişkenleri one-hot encoding değişkenlere çeviriyoruz
X = pd.get_dummies(X, columns=['Gender', 'Previous Curriculum (17/18)2'], drop_first=True)

# Veriyi eğitim (%70), doğrulama (%10) ve test (%20) olarak ayırıyoruz
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

# Sınıf isimlerini alıyoruz (High, Low, Medium)
class_names = np.unique(y)

