import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn_extra.cluster import KMedoids
from scipy.spatial.distance import pdist, squareform

# Génération de données aléatoires
nn = 50
x = np.vstack([
    np.random.normal(0.5, 0.3, (nn, 2)),
    np.random.normal(4, 0.3, (nn, 2)),
    np.random.normal(2, 0.5, (nn, 2))
])

# Affichage initial des données
plt.scatter(x[:, 0], x[:, 1])
plt.title('Distribution initiale')
plt.show()

# K-means
kmeans = KMeans(n_clusters=3)
cl3 = kmeans.fit(x)
plt.scatter(x[:, 0], x[:, 1], c=cl3.labels_)
plt.scatter(cl3.cluster_centers_[:, 0], cl3.cluster_centers_[:, 1], color='yellow', marker='o')
plt.title("Après le k-means")
plt.show()

print(cl3.inertia_)

# K-medoids (PAM)
kmedoids = KMedoids(n_clusters=3, random_state=0).fit(x)
plt.scatter(x[:, 0], x[:, 1], c=kmedoids.labels_)
plt.title("K-medoids (PAM)")
plt.show()

# Détermination du nombre de clusters
K = range(2, 11)
J = []
for k in K:
    kmeans = KMeans(n_clusters=k).fit(x)
    J.append(kmeans.inertia_)

plt.plot(K, J, '-o')
plt.title("Détermination du nombre de clusters")
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertia')
plt.show()

# Clustering hiérarchique
md = pdist(x)
for method in ['ward', 'single', 'complete', 'average', 'centroid', 'median']:
    hh = linkage(md, method=method)
    dendrogram(hh)
    plt.title(f'Clustering hiérarchique avec {method}')
    plt.show()

# DBSCAN
for eps in [0.1, 0.2, 0.3]:
    db = DBSCAN(eps=eps, min_samples=3).fit(x)
    plt.scatter(x[:, 0], x[:, 1], c=db.labels_)
    plt.title(f"DBSCAN pour eps = {eps}")
    plt.show()
