import numpy as np

def k_means(X, K, max_iters=100):
    # Étape 1: Initialisation des centroïdes
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]

    for _ in range(max_iters):
        # Étape 2: Affectation des points aux clusters
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_cluster = np.argmin(distances, axis=0)
        print(closest_cluster)
        # Étape 3: Mise à jour des centroïdes
        new_centroids = np.array([X[closest_cluster==k].mean(axis=0) for k in range(K)])

        # Vérifier la convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return closest_cluster, centroids

# Exemple d'utilisation
X = np.random.rand(100, 2)  # 100 points dans un espace 2D
K = 3  # Nombre de clusters
clusters, centroids = k_means(X, K)

# Affichage des résultats
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, c='red')  # Centroïdes
plt.show()
