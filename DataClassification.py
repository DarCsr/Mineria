import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar datos
df = pd.read_csv('data.csv')

# Filtrar solo 3 países más frecuentes
top_paises = df['Country'].value_counts().nlargest(3).index.tolist()
df = df[df['Country'].isin(top_paises)]

# Eliminar datos vacíos y anómalos
df = df[['Quantity', 'UnitPrice', 'Country']].dropna()
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# Variables
X = df[['Quantity', 'UnitPrice']]  # Características
y = df['Country']                 # Etiqueta (país)

# Codificar etiquetas (de texto a número)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Crear modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predecir
y_pred = knn.predict(X_test)

# Evaluar
print("Accuracy del modelo KNN:", accuracy_score(y_test, y_pred))
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred, target_names=le.classes_))
