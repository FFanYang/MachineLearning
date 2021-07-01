print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question Two task 1 a
from sklearn.preprocessing import MinMaxScaler
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from time import process_time
from sklearn.metrics import davies_bouldin_score
#Loading Data
Path = "dow_jones_index/dow_jones_index.data" # change path
Rawdatas = pandas.read_csv(Path)
array = Rawdatas.values
nrow, ncol = Rawdatas.shape
df = Rawdatas.replace('?',np.nan)
df.dropna(inplace=True)
for columsN in {'open','close','high','low','next_weeks_open','next_weeks_close'}:
    df[columsN] = df[columsN].str.replace(',','').str.replace('$','').astype('float')
df = df.drop(['date'],axis=1)
df = df.drop(['stock'],axis=1)
#  MinMax Scaler implemented
Scal = MinMaxScaler()
Scal.fit(df)
MMS=Scal.transform(df)
#FeatureSelection implement PCA
FSPCA= PCA(n_components=6)
FSPCA.fit(MMS)
DOWJ = FSPCA.transform(MMS)
print(FSPCA.explained_variance_.round(6))
#Implemented Kmesns-algorithms
SSEDistance = []
Ks = range(3,8)

for kss in Ks:
    kmeans = KMeans(n_clusters=kss, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    YKM = kmeans.fit_predict(DOWJ)
    start = process_time()
    SSEDistance.append(kmeans.inertia_)
plt.plot(Ks,SSEDistance)
plt.xlabel('Sum of squares errors')
plt.title('Curve of SSE K point')
plt.show()
Sse_dow_jones = davies_bouldin_score(DOWJ,YKM)
print(kss,'clusters KMeans Time used:', process_time()-start,' seconds', 'sse_dowjonws', Sse_dow_jones )
#CSM PLOT display
K_M = KMeans(n_clusters=6,
            init='k-means++',
            n_init=20,
            max_iter=500,
            random_state=2)
Y_K_M = K_M.fit_predict(DOWJ)
Cl = np.unique(YKM)
Cls = Cl.shape[0]
sil_values = silhouette_samples(DOWJ, YKM, metric='euclidean')
YAXlower, YAXupper = 0, 0
Y_tick = []
for lo, cl in enumerate(Cl):
    C_sil_values = sil_values[YKM == cl]
    C_sil_values.sort()
    YAXupper += len(C_sil_values)
    Colr = cm.jet(float(lo) / Cls)
    plt.barh(range
             (YAXlower, YAXupper),
             C_sil_values,
             height=5.0,
             edgecolor='none',
             color=Colr)
    Y_tick.append((YAXlower + YAXupper) / 2.5)
    YAXlower += len(C_sil_values)
sil_AVG = np.mean(sil_values)
plt.axvline(sil_AVG, color="black", linestyle="--")
plt.suptitle(sil_AVG, fontsize=18)
plt.yticks(Y_tick, Cl + 1)
plt.ylabel('Display Clusters')
plt.xlabel('CSM Display')
print('CSM',sil_AVG)
plt.tight_layout()
#save image when running the code
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/picDowc=5', dpi=300)
plt.show()



