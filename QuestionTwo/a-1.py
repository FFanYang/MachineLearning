#Question 2 task 1 b
from sklearn.preprocessing import MinMaxScaler
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Fan Yang')

import pandas
from IPython.display import Image
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from time import process_time

#Load Data
Path = "Live_20210128.csv"
rawdata = pandas.read_csv(Path)
array = rawdata.values
nrow, ncol = rawdata.shape

df= rawdata



# for col_name in {'open','close','high','low','next_weeks_open','next_weeks_close'}:
#     df[col_name] = df[col_name].str.replace(',','').str.replace('$','').astype('float')
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
#dow_jones = df

#Get sse
tmp_score = -1
tmp_y = None
tmp_n = 0
tmp_e = 0
for eps in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2]:
    for n in [5, 7, 10, 14, 19, 20, 25, 31]:
        db = DBSCAN(eps=eps, min_samples=n, metric='euclidean')
        start = process_time()
        y_db = db.fit_predict(df)
        t = process_time()-start
        clusters = len(np.unique(y_db))
        if (sum(y_db == -1) > 0):
            clusters = clusters - 1
        if (clusters > 1):
            score = silhouette_score(df, y_db)
        else:
            score = -1
        if (score > tmp_score):
            tmp_score = score
            tmp_y = y_db
            tmp_e = eps
            tmp_n = n
        print('score=', score, ',timeused=',t, ',e=', eps, ',n=', n, 'lsd=', sum(y_db == -1), ',clusters=', clusters)
        sse = davies_bouldin_score(df,y_db)
        print('the selected e is ', tmp_e, ',n_samples is ', tmp_n, ',SSE=', sse, ',for its max silhouette avg:', tmp_score)

db = DBSCAN(eps=tmp_e, min_samples=tmp_n, metric='euclidean')
y_km = db.fit_predict(df)
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
plt.xlabel('CSM Display')
plt.suptitle(silhouette_avg, fontsize=16)
plt.tight_layout()
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/piclive2C=5',
    dpi=300)
plt.show()






