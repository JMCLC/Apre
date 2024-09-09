from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.io.arff import loadarff
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

data = loadarff('C:/pd_speech.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
df.head()
X = df.drop('class', axis = 1)
y = df['class']

states = [0,1,2]
minmax = MinMaxScaler()
X = minmax.fit_transform(X)

for i in states:
    cluster = KMeans(n_clusters = 3, random_state = i).fit(X)
    prediction = cluster.predict(X)
    silhouette = silhouette_score(X, prediction)
    contingency = contingency_matrix(y, prediction)
    purity = np.sum(np.amax(contingency, axis = 0)) / np.sum(contingency)
    print("For state =", i, "K-means purity score is", purity, "and silhouette score is", silhouette)
    if i == 0:
        variances = np.var(X, axis=0)
        idx = np.argsort(variances)[::-1]
        X_new = X[:,idx[:2]]
        scatter = plt.scatter(X_new[:,0], X_new[:,1], c=prediction, edgecolor='black')
        plt.legend(*scatter.legend_elements(),loc="upper right", title="Cluster")
        plt.title("K-Means")
        plt.show()
        scatter = plt.scatter(X_new[:,0], X_new[:,1], c=y, edgecolor="black")
        plt.title("Original")
        plt.show()

for i in range(len(y)):
    pca = PCA(n_components = i, svd_solver = "full")
    pca.fit(X)
    var = 0
    for j in pca.explained_variance_ratio_:
        var += j
    if var >= 0.8:
        print(i, "principal components were needed to reach a variability of ", var)
        break
