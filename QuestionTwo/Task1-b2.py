import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sklearn.metrics
from sklearn.metrics import davies_bouldin_score
import matplotlib.mlab as mlab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.preprocessing import scale
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from IPython import get_ipython
import numpy as np
import pandas
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
Path = "dow_jones_index/dow_jones_index.data" # change path
rawdata = pandas.read_csv(Path)
df = rawdata.replace('?',np.nan)
df.drop(columns=['Column1'],inplace=True)
df.dropna(inplace=True)
print(df.head())


scaler = MinMaxScaler()
scaler.fit(df)
mm_scaled=scaler.transform(df)

pca= PCA(n_components=4)
pca.fit(mm_scaled)
dow_jones = pca.transform(mm_scaled)
print(pca.explained_variance_.round(4))


neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(dow_jones)
distances, indices = nbrs.kneighbors(dow_jones)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()



start_time = time.time()
ac =  DBSCAN(eps=0.8, min_samples=6, metric='euclidean')
y_db = ac.fit_predict(dow_jones)
end_time = time.time()
clusters = ac.labels_
plt.scatter(dow_jones[:,0], dow_jones[:,1],c=clusters, cmap='Paired')
plt.title("dbscan")
plt.show()
Time_taken_db = end_time - start_time
sse_dowjones_db = davies_bouldin_score(dow_jones,clusters)
csm_dowjones_db = silhouette_score(dow_jones,clusters)
print(Time_taken_db,sse_dowjones_db,csm_dowjones_db)

#CSM PlOT FOR DBSCAN
ac = DBSCAN(eps=0.8, min_samples=6, metric='euclidean')
y_km = ac.fit_predict(dow_jones)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(dow_jones, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
plt.savefig(r'C:\Users\mumin\OneDrive\Documents\DWcases\waterplot3.png', dpi=300)
plt.show()
