import pandas as pd

df = pd.read_csv('data.csv')

# Estadísticas generales para columnas numéricas
print(df.describe())
print("\n")

# Estadísticas también para columnas no numéricas
print(df.describe(include='all'))
print("\n")

# Agrupar por país
agrupado = df.groupby('Country')[['Quantity', 'UnitPrice']].agg(['mean', 'sum', 'count'])
print(agrupado)
print("\n")

# Agrupar por producto
producto_stats = df.groupby('Description')[['Quantity', 'UnitPrice']].agg(['mean', 'sum']).sort_values(('Quantity', 'sum'), ascending=False)
print(producto_stats.head(10))
print("\n")