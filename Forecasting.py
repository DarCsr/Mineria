import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar datos
df = pd.read_csv('data.csv')

# Convertir columna de fecha a datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Agrupar datos por fecha (ejemplo: suma de Quantity por día)
df_daily = df.groupby(df['InvoiceDate'].dt.date)['Quantity'].sum().reset_index()
df_daily.columns = ['Date', 'Quantity']

# Crear variable numérica para la fecha (día desde el inicio)
df_daily['DateOrdinal'] = pd.to_datetime(df_daily['Date']).map(pd.Timestamp.toordinal)

# Variables independientes y dependientes
X = df_daily[['DateOrdinal']]
y = df_daily['Quantity']

# Crear y entrenar modelo
model = LinearRegression()
model.fit(X, y)

# Predecir para las fechas actuales
y_pred = model.predict(X)

# Predecir para nuevos días (7 días futuros)
last_date = df_daily['Date'].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)

# Convertir a ordinal para la predicción
future_ordinals = future_dates.map(pd.Timestamp.toordinal).to_frame(name='DateOrdinal')

# Predicción futura
future_pred = model.predict(future_ordinals)

# Graficar resultados
plt.figure(figsize=(10,6))
plt.plot(df_daily['Date'], y, label='Datos reales')
plt.plot(df_daily['Date'], y_pred, label='Predicción modelo', linestyle='--')
plt.plot(future_dates, future_pred, label='Predicción futura', linestyle=':')
plt.xlabel('Fecha')
plt.ylabel('Cantidad')
plt.title('Pronóstico de cantidad diaria con regresión lineal')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
