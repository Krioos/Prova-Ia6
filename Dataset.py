from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Feature e target in un unico dataframe
df = pd.concat([X, y], axis=1)

# Dimensioni
print("-------Dimensioni-------")
print(df.shape)

# Prime righe
print("-------Head-------")
print(df.head())

# Ultime righe
print("-------Tail-------")
print(df.tail())

# Informazioni generali
print("-------Info-------")
print(df.info())

# Statistiche descrittive
print("-------Statistiche Descrittive-------")
print(df.describe())

# Outliers
for col in X.columns:
    Q1 = df[col].quantile(0.25)   # Primo Quartile
    Q3 = df[col].quantile(0.75)   # Terzo Quartile
    IQR = Q3 - Q1                 # Interquartile Range
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\nColonna: {col}")
    print(f"Numero di outlier: {len(outliers)}")
    print(outliers[[col]])

# Boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()

# Conteggio delle classi
y_flat = y.iloc[:, 0]
print("Conteggio delle classi:")
print(y_flat.value_counts())

# Grafico distribuzione classi
plt.figure(figsize=(6, 4))
sns.countplot(x=y_flat)
plt.title("Distribuzione delle classi")
plt.xlabel("Classe")
plt.ylabel("Numero di osservazioni")
plt.show()

# -----------------------------------------------
# Heatmap
# -----------------------------------------------
plt.figure(figsize=(8, 6))
correlation_matrix = X.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,          # mostra i valori numerici
    fmt=".2f",           # 2 decimali
    cmap="coolwarm",     # colori: blu=correlazione negativa, rosso=positiva
    vmin=-1, vmax=1,     # scala fissa da -1 a 1
    linewidths=0.5,      # separatori tra celle
    square=True          # celle quadrate
)
plt.title("Heatmap di Correlazione - Iris Features")
plt.tight_layout()
plt.show()

print("\n-------Matrice di Correlazione-------")
print(correlation_matrix)

# -----------------------------------------------
# Encoding del target con LabelEncoder
# -----------------------------------------------
le = LabelEncoder()

# fit_transform: apprende le classi e le converte in interi (0, 1, 2)
y_encoded = le.fit_transform(y_flat)

print("\n-------Encoding del Target-------")
print("Classi originali →", le.classes_)
print("Mapping:")
for i, cls in enumerate(le.classes_):
    print(f"  {cls} → {i}")

print("\nPrime 10 etichette originali:", list(y_flat[:10]))
print("Prime 10 etichette codificate:", list(y_encoded[:10]))

# Aggiunta al dataframe
df["target_encoded"] = y_encoded
print("\nDataframe con target encoded (head):")
print(df.head())

# Verifica: le classi sono bilanciate anche dopo l'encoding?
print("\nDistribuzione target_encoded:")
print(df["target_encoded"].value_counts().sort_index())