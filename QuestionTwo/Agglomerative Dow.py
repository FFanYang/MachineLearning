
print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question Two task 1 c
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import time
from sklearn.preprocessing import MinMaxScaler
#Load Data
Path = "dow_jones_index/dow_jones_index.data" # change path
Rawdatas = pandas.read_csv(Path)
Arrays = Rawdatas.values
nrow, ncol = Rawdatas.shape
df = Rawdatas.replace('?',np.nan)
df.dropna(inplace=True)
for columN in {'open',
                 'close','high','low',
                 'next_weeks_open','next_weeks_close'}:
    df[columN] = df[columN].str.replace(',','').str.replace('$','').astype('float')
df = df.drop(['date'],axis=1)
df = df.drop(['stock'],axis=1)
# MinMax Scaler implemented
Scal = MinMaxScaler()
MMS=Scal.fit_transform(df)
#FeatureSelection (PCA implemented)
FSPCA= PCA(n_components= 9)
FSPCA.fit(MMS)
DJ = FSPCA.transform(MMS)
SSEdistance = []
print(df.shape)

KS = range(3,9)
for loopk in KS:
    KAC = AgglomerativeClustering(n_clusters=loopk,affinity='euclidean', linkage='complete').fit(DJ)
    labels = KAC.labels_
    SSEdistance .append(silhouette_score(DJ,labels))
plt.plot(KS,SSEdistance )
plt.xlabel('Sum of squares errors which mean K')
plt.title('Curve of SSE K point')
plt.show()
STime = time.time()
AggClu = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
labels = AggClu.fit_predict(DJ)
ETime = time.time()
Time_taken_ag = ETime - STime
SSE_DJ_AVG= davies_bouldin_score(DJ,labels)
CSM_DJ_AVG = silhouette_score(DJ,labels)
print('CSM',CSM_DJ_AVG,'clusters KMeans time used',Time_taken_ag,'sum of Squares errors',SSE_DJ_AVG)

#CSM PLOT
AggCl = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete'  )
YKM = AggCl.fit_predict(DJ)
CLabel = np.unique(YKM)
Clu_N = CLabel.shape[0]
CSil_values = silhouette_samples(DJ, YKM, metric='euclidean')
Yaxlow, Yaxup = 0, 0
YT = []
for loop, loopcl in enumerate(CLabel):
    Sil_C_values =CSil_values[YKM == loopcl]
    Sil_C_values.sort()
    Yaxup += len(Sil_C_values)
    color = cm.jet(float(loop) / Clu_N)
    plt.barh(range(Yaxlow, Yaxup), Sil_C_values, height=5.0, edgecolor='none', color=color)
    YT.append((Yaxlow + Yaxup) / 2.5)
    Yaxlow += len(Sil_C_values)
CSil_V_AVG = np.mean(CSil_values)
plt.axvline(CSil_V_AVG, color="black", linestyle="-")
plt.suptitle(CSil_V_AVG, fontsize=18)
plt.yticks(YT, CLabel + 1)
plt.ylabel('Display Clusters')
plt.xlabel('CSM DISPLAY')
plt.tight_layout()
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/picSalesc=3', dpi=300)
plt.show()


