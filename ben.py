import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsClassifier

# Veri kümesini yükleyelim
database = load_diabetes()

# Kümeyi DataFrame'e çevirelim
df = pd.DataFrame(data=database.data, columns=database.feature_names)
df['target'] = database.target

print("\nLineer Regresyon Veri Seti:")
print(df.head())

# ----------------- REGRESYON BÖLÜMÜ -----------------

# Basit Lineer Regresyon
x = df[['bmi']]
y = df['target']

x_test, x_train, y_test, y_train = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nEğitim Seti Boyutu: {x_train.shape}")
print(f"Test Seti Boyutu: {x_test.shape}")

print("\nBasit Lineer Regresyon Modeli Eğitiliyor...")

model = LinearRegression()
model.fit(x_train, y_train)

print("Basit Lineer Regresyon Modeli Eğitildi!\n")
print("Katsayısal:")
print(model.coef_)
print(f"Intercept (b0) : {model.intercept_}")

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f"\nBasit Lineer Regresyon r2 skoru: {r2:.4f}")

# Çoklu Lineer Regresyon
print("\n------------------------------------------------------------------------------")
print("\nÇoklu Lineer Regresyon\n")

new_x = df[database.feature_names]
new_y = df['target']

new_x_test, new_x_train, new_y_test, new_y_train = train_test_split(new_x, new_y, test_size=0.2, random_state=42)

print(f"Çoklu Lineer Regresyon Eğitim Seti Boyutu: {new_x_train.shape}")
print(f"Çoklu Lineer Regresyon Test Seti Boyutu: {new_x_test.shape}")

print("\nÇoklu Lineer Regresyon Modeli Eğitiliyor...")

model2 = LinearRegression()
model2.fit(new_x_train, new_y_train)
print("Çoklu Lineer Regresyon Modeli Eğitildi!\n")
print("Katsayısal:")
print(model2.coef_)
print(f"Intercept (b0) : {model2.intercept_}")

y_pred1 = model2.predict(new_x_test)
r2 = r2_score(new_y_test, y_pred1)
print(f"\nÇoklu Lineer Regresyon r2 skoru = {r2:.4f}")

# Hata Metrikleri
print("\nBasit ve Çoklu Lineer Regresyon Hata Metrikleri ")
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nBasit Lineer Regresyon MSE: {mse:.4f}")
print(f"Basit Lineer Regresyon MAE: {mae:.4f}")

mse1 = mean_squared_error(new_y_test, y_pred1)
mae1 = mean_absolute_error(new_y_test, y_pred1)
print(f"Çoklu Lineer Regresyon MSE: {mse1:.4f}")
print(f"Çoklu Lineer Regresyon MAE: {mae1:.4f}")

# ----------------- KNN SINIFLANDIRMA BÖLÜMÜ -----------------

print("\n------------------------------------------------------------------------------")
print("\nKNN SINIFLANDIRMA MODELİ\n")

# Hedef değişkeni kategorilere ayır (ortalamaya göre binarize et)
df['target_class'] = (df['target'] > df['target'].mean()).astype(int)

# Özellikler ve hedef sınıf
X = df[database.feature_names]
y_class = df['target_class']

# Eğitim ve test verisi
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# KNN Modeli
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_class, y_train_class)

# Tahmin
y_pred_knn = model_knn.predict(X_test_class)

# Sonuçlar
print("KNN Modeli Eğitildi ve Test Verisi ile Tahmin Yapıldı.\n")

print("KNN Doğruluk (Accuracy):", accuracy_score(y_test_class, y_pred_knn))
print("\nKarmaşıklık Matrisi (Confusion Matrix):\n", confusion_matrix(y_test_class, y_pred_knn))
print("\nSınıflandırma Raporu (Classification Report):\n", classification_report(y_test_class, y_pred_knn))
