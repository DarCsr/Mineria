import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('data.csv')

text = " ".join(str(desc) for desc in df['Description'].dropna())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de palabras - Descripción de productos')
plt.show()
