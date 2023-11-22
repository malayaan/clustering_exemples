import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Résumé statistique
print(iris_df.describe())

# Affichage des premières lignes
print(iris_df.head())

# Pairplot pour visualiser les relations entre les caractéristiques
sns.pairplot(iris_df, hue='species')
plt.show()

# Histogrammes pour chaque caractéristique
iris_df.drop('species', axis=1).hist(bins=20)
plt.show()
