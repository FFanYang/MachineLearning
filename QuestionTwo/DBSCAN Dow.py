print('Student Name: Fan Yang '
      'Student ID: 20104813')
#Question Two task 1 b
import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from time import process_time
from sklearn.preprocessing import MinMaxScaler
#Load Data
Path = "dow_jones_index/dow_jones_index.data" # change path
Rawdatas = pandas.read_csv(Path)
Arrays = Rawdatas.values
nrow, ncol = Rawdatas.shape
df = Rawdatas.replace('?',np.nan)
df.dropna(inplace=True)
for columsN in {'open',
                 'close','high','low',
                 'next_weeks_open','next_weeks_close'}:
    df[columsN] = df[columsN].str.replace(',','').str.replace('$','').astype('float')
df = df.drop(['date'],axis=1)
df = df.drop(['stock'],axis=1)

# MinMax Scaler implemented
Scal = MinMaxScaler()
MMS=Scal.fit_transform(df)
#FeatureSelection (PCA implemented)
FSPCA= PCA(n_components= 9)
FSPCA.fit(MMS)
df = FSPCA.transform(MMS)
print(df.shape)
#GETTING SUM OF SQUARES ERRORS
tscore = -1
t_y = None
t_N = 0
t_E = 0
#Loop to pick up parameters and get the best result
for leps in [0.25,0.3,0.45,0.75,1.5,0.5]:# If set parameter as 0.75, after the highest will get.
    for n in [1,2,7,10,14,19,25,31]:
        DBS = DBSCAN(eps=leps, min_samples=n, metric='euclidean')
        start = process_time()
        YDB = DBS.fit_predict(df)
        print(YDB)
        Times = process_time()-start
        clu = len(np.unique(YDB))
        if (sum(YDB == -2) > 0):
            clus = clu - 2
        if (clu > 3):
            Result = silhouette_score(df, YDB)
        else:
            Result  = -1
        if (Result  > tscore):
            tscore = Result
            t_E = leps
            t_N = n
        print(',Time_Taken=',Times,
              ',Clusters=', clu)
        if(clu>1):
            SSE = davies_bouldin_score(df,YDB)
        else:
            SSE = 0
        print('Sum of squares errors =', SSE,
              ', silhouette:', tscore)

#DBSCAN ALGORITHM
datebase = DBSCAN(eps=t_E, min_samples=t_N,metric='euclidean')
YKM = datebase.fit_predict(df)
C_label = np.unique(YKM)
Clusters_N = C_label.shape[0]
silhouette_valuess = silhouette_samples(df, YKM, metric='euclidean')
YaxLow, YaxUp = 0, 0
yticks = []
for lo, cl in enumerate(C_label):
    Cl_sil_values = silhouette_valuess[YKM == cl]
    Cl_sil_values.sort()
    YaxUp += len(Cl_sil_values)
    Col = cm.jet(float(lo) / Clusters_N)
    plt.barh(range(YaxLow, YaxUp), Cl_sil_values, height=5.0, edgecolor='none', color=Col)
    yticks.append((YaxLow + YaxUp) / 2.5)
    YaxLow += len(Cl_sil_values)
sil_AVG = np.mean(silhouette_valuess)
plt.axvline(sil_AVG, color="black", linestyle="-")
plt.yticks(yticks, C_label + 1)
plt.ylabel('Displays number of Clusters')
plt.xlabel('CSM Display')
plt.suptitle(sil_AVG, fontsize=18)
plt.tight_layout()
plt.savefig('/Users/fanyang/OneDrive - AUT University/MASTER OF AUT 20 July 2020/Semester 1 Middle 1 March  2021/Data mining Machine learning/Project/AssignmentTwo/figure/piclive2C=5',
    dpi=500)
plt.show()
