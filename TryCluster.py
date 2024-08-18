import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing CSV files
data_dir = 'data/individual_stocks_5yr'

# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to extract features from the time series data
def extract_features(df):
    features = {}
    close_prices = df['close']
    high_prices = df['high']
    low_prices = df['low']
    volume = df['volume']

    # Calculate daily returns
    features['mean_return'] = close_prices.pct_change().mean()
    features['volatility'] = close_prices.pct_change().std()

    # Relative Strength Index (RSI)
    delta = close_prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    features['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]

    # Moving Average Convergence Divergence (MACD)
    exp1 = close_prices.ewm(span=12, adjust=False).mean()
    exp2 = close_prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    features['macd'] = macd.iloc[-1]
    features['macd_signal'] = macd.ewm(span=9, adjust=False).mean().iloc[-1]

    # Additional Moving Averages
    features['moving_average_50'] = close_prices.rolling(window=50).mean().iloc[-1]
    features['moving_average_200'] = close_prices.rolling(window=200).mean().iloc[-1]

    # Bollinger Bands
    rolling_mean = close_prices.rolling(window=20).mean()
    rolling_std = close_prices.rolling(window=20).std()
    features['bollinger_upper'] = (rolling_mean + 2 * rolling_std).iloc[-1]
    features['bollinger_lower'] = (rolling_mean - 2 * rolling_std).iloc[-1]

    # Average True Range (ATR)
    high_low = high_prices - low_prices
    high_close = np.abs(high_prices - close_prices.shift())
    low_close = np.abs(low_prices - close_prices.shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    features['atr'] = tr.rolling(window=14).mean().iloc[-1]

    # Trading Volume
    features['avg_volume'] = volume.rolling(window=50).mean().iloc[-1]

    return pd.Series(features)

# Load all CSV files and extract features
all_features = []
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)
        df = load_data(file_path)
        features = extract_features(df)
        all_features.append(features)

features_df = pd.DataFrame(all_features)
features_df.fillna(features_df.mean(), inplace=True)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)
n_components = min(10, scaled_features.shape[1], scaled_features.shape[0])
pca = PCA(n_components=n_components)  # Adjust the number of components as needed
pca_features = pca.fit_transform(scaled_features)

# K-Means Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans_clusters = kmeans.fit_predict(pca_features)
features_df['KMeans_Cluster'] = kmeans_clusters
kmeans_silhouette = silhouette_score(pca_features, kmeans_clusters)
print(f'K-Means Silhouette Score: {kmeans_silhouette}')

# Gaussian Mixture Models
gmm = GaussianMixture(n_components=10, random_state=42)
gmm_clusters = gmm.fit_predict(pca_features)
features_df['GMM_Cluster'] = gmm_clusters
gmm_silhouette = silhouette_score(pca_features, gmm_clusters)
print(f'GMM Silhouette Score: {gmm_silhouette}')

# Agglomerative Clustering
agglomerative = AgglomerativeClustering(n_clusters=10)
agglo_clusters = agglomerative.fit_predict(pca_features)
features_df['Agglomerative_Cluster'] = agglo_clusters
agglo_silhouette = silhouette_score(pca_features, agglo_clusters)
print(f'Agglomerative Clustering Silhouette Score: {agglo_silhouette}')

# BIRCH Clustering
birch = Birch(n_clusters=10)
birch_clusters = birch.fit_predict(pca_features)
features_df['BIRCH_Cluster'] = birch_clusters
birch_silhouette = silhouette_score(pca_features, birch_clusters)
print(f'BIRCH Clustering Silhouette Score: {birch_silhouette}')

# Visualize the clusters using t-SNE with adjusted parameters
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
tsne_results = tsne.fit_transform(pca_features)

features_df['tsne-2d-one'] = tsne_results[:,0]
features_df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(20, 20))

# K-Means Clusters
plt.subplot(2, 2, 1)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="KMeans_Cluster",
    palette=sns.color_palette("hsv", 10),
    data=features_df,
    legend="full",
    alpha=0.6
)
plt.title('K-Means Clustering Visualization with t-SNE')

# GMM Clusters
plt.subplot(2, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="GMM_Cluster",
    palette=sns.color_palette("hsv", 10),
    data=features_df,
    legend="full",
    alpha=0.6
)
plt.title('GMM Clustering Visualization with t-SNE')

# Agglomerative Clusters
plt.subplot(2, 2, 3)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Agglomerative_Cluster",
    palette=sns.color_palette("hsv", 10),
    data=features_df,
    legend="full",
    alpha=0.6
)
plt.title('Agglomerative Clustering Visualization with t-SNE')

# BIRCH Clusters
plt.subplot(2, 2, 4)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="BIRCH_Cluster",
    palette=sns.color_palette("hsv", 10),
    data=features_df,
    legend="full",
    alpha=0.6
)
plt.title('BIRCH Clustering Visualization with t-SNE')

plt.tight_layout()
plt.show()
