import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar los datos
df = pd.read_csv('data.csv')

# Asegurar que sean valores numéricos válidos
df = df[['Quantity', 'UnitPrice']].dropna()
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# Correlación
correlacion = df['Quantity'].corr(df['UnitPrice'])
print(f"Correlación entre cantidad y precio unitario: {correlacion:.4f}")

# Variables
X = df[['Quantity']]   # independiente
y = df['UnitPrice']    # dependiente

# Crear modelo lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción
y_pred = modelo.predict(X)

# R^2 score
r2 = r2_score(y, y_pred)
print(f"Coeficiente de determinación R²: {r2:.4f}")

# Gráfico de regresión
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Quantity'], y=df['UnitPrice'], alpha=0.3, label='Datos reales')
plt.plot(df['Quantity'], y_pred, color='red', label='Regresión lineal')
plt.xlabel('Cantidad')
plt.ylabel('Precio Unitario')
plt.title('Regresión lineal: Precio vs Cantidad')
plt.legend()
plt.show()
