import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('data.csv')

# Verifica columnas numéricas disponibles
print(df.select_dtypes(include='number').columns)

# Pie Chart – Distribución por país (top 5)
top_paises = df['Country'].value_counts().nlargest(5)
top_paises.plot.pie(autopct='%1.1f%%')
plt.title('Distribución por País (Top 5)')
plt.ylabel('')
plt.show()

# Histogram – Cantidad de productos por factura
plt.hist(df['Quantity'], bins=30)
plt.title('Histograma de Cantidad')
plt.xlabel('Cantidad')
plt.ylabel('Frecuencia')
plt.show()

# Boxplot – Precio unitario
sns.boxplot(x=df['UnitPrice'])
plt.title('Boxplot de Precios Unitarios')
plt.show()

# Line Plot – Ventas en el tiempo (suma por fecha)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
ventas_diarias = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum()
ventas_diarias.plot()
plt.title('Ventas por Día')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Total')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot – Cantidad vs Precio
plt.scatter(df['Quantity'], df['UnitPrice'], alpha=0.5)
plt.title('Cantidad vs Precio Unitario')
plt.xlabel('Cantidad')
plt.ylabel('Precio Unitario')
plt.show()

# Generar automáticamente histogramas para todas las columnas numéricas
numericas = df.select_dtypes(include='number')
for col in numericas.columns:
    plt.hist(df[col], bins=30)
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()
