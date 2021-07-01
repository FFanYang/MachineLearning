# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Fan Yang')

import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from time import process_time
from sklearn.metrics import davies_bouldin_score
#Load Data
Path = "basket_analysis.csv"
df = pandas.read_csv(Path)
array = df.values
nrow, ncol = df.shape


#scaling using MinMax Scaler Normalizw
scaler = MinMaxScaler()
scaler.fit(df)
mm_scaled=scaler.transform(df)

#PCA FeatureSelection Dimension deduce
pca= PCA(n_components=4)
pca.fit(mm_scaled)
dow_jones = pca.transform(mm_scaled)
print(pca.explained_variance_.round(4))

#Kmesns
Sum_of_sqared_distances = []
K = range(1,9)
for k in K:
    km = KMeans(n_clusters=k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(dow_jones)
    start = process_time()
    Sum_of_sqared_distances.append(km.inertia_)
plt.plot(K,Sum_of_sqared_distances)
plt.xlabel('SSE')
plt.title('Curve of SSE K point')
plt.show()
Sse_dow_jones = davies_bouldin_score(dow_jones,y_km)
print(k,'clusters KMeans time used:', process_time()-start,' seconds', 'sse_dowjonws', Sse_dow_jones )


#CSM PLOT



km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(dow_jones)


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
plt.suptitle(silhouette_avg, fontsize=16)
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('CSM Display')
print('CSM',silhouette_avg)

plt.tight_layout()
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/picSalesc=3', dpi=300)
plt.show()



