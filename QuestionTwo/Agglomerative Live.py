#Question 2 task 1 b
from sklearn.preprocessing import MinMaxScaler
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Fan Yang')

import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import time
#Load Data
Path = "Live_20210128.csv"
rawdata = pandas.read_csv(Path)
array = rawdata.values
nrow, ncol = rawdata.shape

df= rawdata
df = df.drop(['status_id'],axis=1)
df = df.drop(['status_published'],axis=1)
df = df.drop(['status_type'],axis=1)
df = df.drop(['Column1', 'Column2','Column3','Column4'],axis=1)
#scaling using MinMax Scaler
scaler = MinMaxScaler()
mm_scaled=scaler.fit_transform(df)
#PCA FeatureSelection
pca= PCA(n_components=8)
pca.fit(mm_scaled)
dow_jones = pca.transform(mm_scaled)
print(pca.explained_variance_.round(4))
print(df)

Sum_of_sqared_distances = []
K = range(2,9)
for k in K:
    ac = AgglomerativeClustering(n_clusters=k,affinity='euclidean', linkage='complete').fit(dow_jones)
    labels = ac.labels_
    Sum_of_sqared_distances.append(silhouette_score(dow_jones,labels))
plt.plot(K,Sum_of_sqared_distances)
plt.xlabel('K')
plt.xlabel('SSE')
plt.title('Curve of SSE K point')
plt.show()

start_time = time.time()
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
labels = ac.fit_predict(dow_jones)
end_time = time.time()
Time_taken_ag = end_time - start_time
sse_dowjones_ag = davies_bouldin_score(dow_jones,labels)
csm_dowjones_ag = silhouette_score(dow_jones,labels)
print('clusters KMeans time used',Time_taken_ag,'sum of Squares errors',sse_dowjones_ag,'CSM',csm_dowjones_ag)

#CSM PLOT
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
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
plt.suptitle(silhouette_avg, fontsize=16)
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('CSM DISPLAY')

plt.tight_layout()
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/picSalesc=3', dpi=300)
plt.show()


