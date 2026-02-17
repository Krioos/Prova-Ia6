from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
    Q1 = df[col].quantile(0.25) # Primo Quartile
    Q3 = df[col].quantile(0.75) # Terzo Quartile
    IQR = Q3 - Q1 # Interquartile Range

    lower = Q1 - 1.5 * IQR 
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nColonna: {col}")
    print(f"Numero di outlier: {len(outliers)}")
    print(outliers[[col]])

# Boxplots
plt.figure(figsize=(12, 8))
for i, col in enumerate(X.columns, 1):
    plt.subplot(2, 2, i)   # 2 righe, 2 colonne (per 4 feature)
    sns.boxplot(y=df[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Conteggio delle classi
print("Distribuzione delle classi:")
print(y.value_counts())

y_flat = y.iloc[:, 0]

# Ora y_flat è 1D e funziona con pandas e seaborn
print("Conteggio delle classi:")
print(y_flat.value_counts())

# Grafico
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x=y_flat)
plt.title("Distribuzione delle classi")
plt.xlabel("Classe")
plt.ylabel("Numero di osservazioni")
plt.show()