import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Charger les données
data = pd.read_csv('data/data.csv', header=None)
column_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data.columns = column_names

# Conversion des labels
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# Normalisation des données
features = data.drop(columns=['id', 'diagnosis'])
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
features_df = pd.DataFrame(features_scaled, columns=features.columns)

# Séparer les données bénignes et malignes
benign_data = features_df[data['diagnosis'] == 0]
malignant_data = features_df[data['diagnosis'] == 1]

# Histogrammes des caractéristiques
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 20))
axes = axes.ravel()
for idx, col in enumerate(features_df.columns):
    axes[idx].hist(benign_data[col], bins=20, alpha=0.5, label='Benign', color='blue')
    axes[idx].hist(malignant_data[col], bins=20, alpha=0.5, label='Malignant', color='red')
    axes[idx].set_title(col)
    axes[idx].legend(loc='upper right')
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(data, vars=features.columns[:10], hue='diagnosis', palette={0: 'blue', 1: 'red'})
plt.show()

# Réduction de la dimensionnalité avec PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features_scaled)
pca_df = pd.DataFrame(data={'pca1': pca_result[:, 0], 'pca2': pca_result[:, 1], 'diagnosis': data['diagnosis']})

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='diagnosis', data=pca_df, palette={0: 'blue', 1: 'red'}, alpha=0.7)
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()