import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data=pd.read_csv("md_for_Python.csv", delimiter=";")
print(data)


# Sélectionner les colonnes pertinentes pour le clustering (à adapter selon votre cas)
X = data[['top.i', 'bottom.i', 'top.c', 'bottom.c', 'top.pm', 'bottom.pm', 'top.m', 'bottom.m']]

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2)  # Paramètres à ajuster selon votre jeu de données
labels = dbscan.fit_predict(X_scaled)

# Ajouter les résultats au DataFrame
data['cluster'] = labels

# Afficher les résultats
print(data)

# Visualiser les clusters (exemple avec deux dimensions)
plt.scatter(data['top.i'], data['bottom.i'], c=labels, cmap='viridis')
plt.title('DBSCAN Clustering')
plt.xlabel('top.i')
plt.ylabel('bottom.i')
plt.show()