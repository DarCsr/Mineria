import pandas as pd
from scipy.stats import f_oneway, ttest_ind, kruskal

df = pd.read_csv('data.csv')

df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

top_paises = df['Country'].value_counts().nlargest(3).index.tolist()

df_filtrado = df[df['Country'].isin(top_paises)]

grupo1 = df_filtrado[df_filtrado['Country'] == top_paises[0]]['UnitPrice'].dropna()
grupo2 = df_filtrado[df_filtrado['Country'] == top_paises[1]]['UnitPrice'].dropna()
grupo3 = df_filtrado[df_filtrado['Country'] == top_paises[2]]['UnitPrice'].dropna()

# ANOVA
anova_resultado = f_oneway(grupo1, grupo2, grupo3)
print("Resultado ANOVA:")
print(anova_resultado)

# Kruskal-Wallis (alternativa si los datos no son normales)
kruskal_resultado = kruskal(grupo1, grupo2, grupo3)
print("\nResultado Kruskal-Wallis:")
print(kruskal_resultado)

# T-test entre los dos primeros grupos (opcional)
t_test = ttest_ind(grupo1, grupo2)
print("\nResultado T-Test (entre los dos primeros países):")
print(t_test)
