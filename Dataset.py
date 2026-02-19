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



#CODICE SCRITTO PER DECISION TREE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import plot_tree

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#1)importo dataset Iris
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables) 

# Uniamo X e y in un unico DataFrame chiamato 'df'
df = pd.concat([X, y], axis=1)

#2)Inizio l'EDA(Analisi Esplorativa dei Dati)
#1. Visualizzo le parti che compongono il dataset
'''
print(df.shape)
print(df.describe())
print(df.columns) # Controlliamo il nome esatto delle colonne
print(df.isnull().sum())#Non dovrebbero esserci valori null nei dati
'''
# 2. Heatmap di correlazione
#verifico come le varie variabili sono correlate tra di loro
#Serve a Identificare le variabili più importanti
#Scoprire possibili Ridondanze
#Individuare relazioni inaspettate
'''
plt.figure(figsize=(8,6))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title("Correlazione tra le caratteristiche dell'Iris")
plt.show()
'''
'''0: non ci sono correlazioni, Tendente a 1: Correlazione Positiva Perfetta(se uno aumenta l'alto sale in modo proporzionale), 
se tende a -1: Correlazione Negativa Perfetta, (se uno sale l'altra variabile scende in maniera proporzionale))'''

#3.1 Istogramma 
#distribuzione lunghezza petali
'''
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='petal length', kde=True, hue='class', element="step")
plt.title("Distribuzione della Lunghezza dei Petali per Specie")
plt.xlabel("Lunghezza Petalo (cm)")
plt.ylabel("Frequenza")
plt.show()
'''
#distribuzione lunghezza sepali
'''
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='sepal length', kde=True, hue='class', element="step")
plt.title("Distribuzione della Lunghezza dei Sepali per Specie")
plt.xlabel("Lunghezza Sepale(cm)")
plt.ylabel("Frequenza")
plt.show()
'''

#unisco i due grafici in un unico grafico
#1 riga  e 2 colonne
'''
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(data=df, x='petal length', kde=True, hue='class', element="step", ax=axes[0])
axes[0].set_title("Distribuzione della Lunghezza dei Petali per Specie")

sns.histplot(data=df, x='sepal length', kde=True, hue='class', element="step", ax= axes[1])
axes[1].set_title("Distribuzione della Lunghezza dei Sepali per Specie")

#evitiamo sovrapposizioni
plt.tight_layout()
'''
#3.2 Boxplot
'''
# Boxplot per confrontare la larghezza dei sepali tra le varie specie
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='class', y='sepal width', palette="Set2")
plt.title("Confronto Larghezza Sepali tra le Specie")
plt.show()

# Boxplot per confrontare la larghezza dei petali tra le varie specie
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='class', y='petal width', palette="Set2")
plt.title("Confronto Larghezza Sepali tra le Specie")
plt.show()
'''
#unisco i due grafici in un unico grafico
'''
#1 riga  e 2 colonne
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.boxplot(data=df, x='class', y='sepal width', palette="Set2", ax=axes[0])
axes[0].set_title("Confronto Larghezza Sepali")

sns.boxplot(data=df, x='class', y='petal width', palette="Set2", ax=axes[1])
axes[1].set_title("Confronto Larghezza Petali")

#evitiamo sovrapposizioni
plt.tight_layout()
plt.show()  
'''
#3)Fare in preprocessing e imputazione dei dati mancanti (Missing Value)
"Non ci sono dati mancanti nel dataset"

#Suddivido dati per training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Definisco modello 
tree_model=  DecisionTreeClassifier(random_state=42)

#Valutazione modello
tree_scores = cross_val_score(tree_model, X_train, y_train.values.ravel(), cv=5)
#cross_val_score permette di capire quanto il modello sia stabile "simulando" più test all'interno dei dati che già possiedi.
#aggiunta di .values.ravel() perché molte funzioni di Scikit Learn preferiscono un array piatto(serve ad evitare i warning)

#addestramento su tutto il dataset
tree_model.fit(X_train, y_train.values.ravel())

#prova sul 30%, il test set
test_accuracy = tree_model.score(X_test, y_test)

print(f"Accuratezza media in Cross-Validation: {tree_scores.mean():.4f}")
print(f"Accuratezza finale sul Test Set: {test_accuracy:.4f}")
