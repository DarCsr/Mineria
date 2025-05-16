import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Cargar datos
df = pd.read_csv('data.csv')

# Seleccionar variables numéricas y filtrar valores válidos
df = df[['Quantity', 'UnitPrice']].dropna()
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Crear modelo KMeans (definir k clusters)
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df)

# Añadir columna con etiquetas de cluster
df['cluster'] = kmeans.labels_

# Visualizar clusters
plt.figure(figsize=(8,5))
colors = ['red', 'green', 'blue']
for i in range(k):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['Quantity'], cluster_data['UnitPrice'], 
                s=20, color=colors[i], label=f'Cluster {i}')
    
# Centroides
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=100, color='yellow', marker='X', label='Centroides')

plt.xlabel('Cantidad')
plt.ylabel('Precio Unitario')
plt.title('Clusters usando K-Means')
plt.legend()
plt.show()
