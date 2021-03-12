#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import metrics,preprocessing
from sklearn.cluster import KMeans,DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn import metrics,preprocessing
from sklearn import mixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
get_ipython().run_line_magic('matplotlib', 'inline')
import os 


# In[2]:


#各路徑處理


# In[3]:


print(os.getcwd())

#direct to 財管可活化流失客模型(交接)的位置
os.chdir('..')

print(os.getcwd())


# In[4]:


#####原始資料#####


# In[5]:


cust_all = pd.read_csv('建模原始資料\\CUST_ALL1008.csv',engine = 'python')

status = ['活躍客','流失客']
cust_all_alive_lose = cust_all[cust_all.STATUS17_18.isin(status)]

pd.set_option('display.max_columns',None)
print(cust_all_alive_lose.info())
print(cust_all_alive_lose.head(10))


# In[6]:


cust_all2 = cust_all_alive_lose[[
        'ID','R_MF2','F_MF','M_MF','R_BD2','F_BD','M_BD',
        'R_ETF2','F_ETF','M_ETF','R_SI2','F_SI','M_SI',
        'R_INS2','F_INS','M_INS']]

cust_all_var = cust_all_alive_lose[[
        'R_MF2','R_BD2','R_ETF2','R_INS2','R_SI2',
        'F_MF','F_BD','F_ETF','F_SI','F_INS',
        'M_MF','M_BD','M_ETF','M_SI','M_INS']]

print(cust_all_var.head(10))


# In[7]:


#####資料描述性#####


# In[8]:


#####資料清洗#####


# In[9]:


####遺失值處理(F,M的用0填補(NaN->0)、R用999填補(NaN->999))
cust_all_var_r = cust_all_var[['R_MF2','R_BD2','R_ETF2','R_INS2','R_SI2']]

cust_all_var_r2 = cust_all_var_r.replace(to_replace = np.nan,value = 999)

cust_all_var_fm = cust_all_var[['F_MF','F_BD','F_ETF','F_SI','F_INS',
                                'M_MF','M_BD','M_ETF','M_SI','M_INS']]

cust_all_var_fm2 = cust_all_var_fm.replace(to_replace = np.nan,value = 0)

cust_all_var2 = pd.concat([cust_all_var_r2,cust_all_var_fm2],axis = 1)

print(cust_all_var2.head(10))


# In[10]:


#min_max_scaler標準化
min_max_scaler = preprocessing.MinMaxScaler()
cust_all_var2_scale = min_max_scaler.fit_transform(cust_all_var2)

cust_all_var2_scale_df = pd.DataFrame(cust_all_var2_scale)
print(cust_all_var2_scale_df.head(10))


#用平均數與標準差做標準化
scaler_standard = preprocessing.StandardScaler()
cust_all_var2_scale2 = scaler_standard.fit_transform(cust_all_var2)

cust_all_var2_scale2_df = pd.DataFrame(cust_all_var2_scale2)
print(cust_all_var2_scale2_df.head(10))


# In[11]:


#####資料建模#####


# In[12]:


####kmeans


# In[13]:


#kmeans(用平均數與變異數做標準化的)


# In[14]:


#elbow method找出最佳群數(k-means++)
inert = []

for i in range(1,20):
    inert.append(KMeans(n_clusters=i,random_state=10,init = "k-means++").fit(cust_all_var2_scale2_df).inertia_)

print(inert)

sns.lineplot(range(1,20),inert)


# In[15]:


#silhouette method
sil_score_agglo = []

for n_clusters in range(2,20):
    kmeans_model = KMeans(n_clusters=n_clusters,random_state=10,init = "k-means++")
    
    labels = []
    labels = kmeans_model.fit_predict(cust_all_var2_scale2_df)
    sil_score_agglo.append(metrics.silhouette_score(cust_all_var2_scale2_df,labels,metric = 'euclidean'))
    print(sil_score_agglo)
    
sns.lineplot(list(range(2,20)),sil_score_agglo)


# In[ ]:





# In[16]:


#kmeans(用min_max做標準化的)


# In[17]:


#elbow method找出最佳群數(k-means++)
inert = []

for i in range(1,20):
    inert.append(KMeans(n_clusters=i,random_state=10,init = "k-means++").fit(cust_all_var2_scale_df).inertia_)

print(inert)#最後kmeans分群法
kmeans_model = KMeans(n_clusters = 5,random_state = 10,init = "k-means++").fit(cust_all_var2_scale_df)

cust_all_alive_lose['kmeans_cluster'] = kmeans_model.labels_

print(cust_all_alive_lose['kmeans_cluster'].value_counts())

#cust_all_alive_lose.to_csv('客戶資料分群後資料.csv')

sns.lineplot(range(1,20),inert)


# In[18]:


#silhouette method
sil_score_agglo = []

for n_clusters in range(2,20):
    kmeans_model = KMeans(n_clusters=n_clusters,random_state=10,init = "k-means++")
    
    labels = []
    labels = kmeans_model.fit_predict(cust_all_var2_scale_df)
    sil_score_agglo.append(metrics.silhouette_score(cust_all_var2_scale_df,labels,metric = 'euclidean'))
    print(sil_score_agglo)
    
sns.lineplot(list(range(2,20)),sil_score_agglo)


# In[19]:


#kmeans最終結果：
#1.用minmax做標準化的效果相對較好，silhouette最高有到0.85
#2.分群的sse也明顯比直接做標準化小很多。
#3.最終決定分9群，且用minmax標準化方法做


# In[20]:


#最後分群模型
kmeans_model = KMeans(n_clusters = 9,random_state = 10,init = "k-means++").fit(cust_all_var2_scale_df)

cust_all_var2['kmeans_cluster'] = kmeans_model.fit_predict(cust_all_var2_scale_df)
cust_all_alive_lose['kmeans_cluster'] = kmeans_model.labels_
kmeans_initial_points = kmeans_model.cluster_centers_

np.save('建模中間資料\\初始模型\\kmeans_init',kmeans_initial_points)

#print(kmeans_model.cluster_centers_)

print(cust_all_alive_lose['kmeans_cluster'].value_counts())

#cust_all_alive_lose.to_csv('客戶資料分群後資料.csv')


# In[ ]:


####DBSCAN


# In[21]:


dbscan_model = DBSCAN(min_samples = 75,
                      eps=0.05,
                      algorithm = 'kd_tree').fit(cust_all_var2_scale_df)

cust_all_var2['dbscan_cluster'] = dbscan_model.fit_predict(cust_all_var2_scale_df)
cust_all_alive_lose['dbscan_cluster'] = dbscan_model.fit_predict(cust_all_var2_scale_df)

print(cust_all_alive_lose['dbscan_cluster'].value_counts())


# In[ ]:





# In[22]:


####階層式分群(歐基里得距離)


# In[23]:


#找出最佳分群：euclidean & ward
sil_score_agglo = []

for n_clusters in range(2,20):
    hierarchical_model = AgglomerativeClustering(n_clusters = n_clusters,
                                                affinity = 'euclidean',
                                                linkage = 'ward')
    labels = []
    labels = hierarchical_model.fit_predict(cust_all_var2_scale_df)
    sil_score_agglo.append(metrics.silhouette_score(cust_all_var2_scale_df,labels,metric = 'euclidean'))
    print(sil_score_agglo)
    
sns.lineplot(list(range(2,20)),sil_score_agglo)


# In[24]:


#階層式分群(euclidean)
#分10群結果最好


# In[25]:


#建立最佳分群的階層式分群模型
hierarchical_model_euc = AgglomerativeClustering(n_clusters = 10,
                                                affinity = 'euclidean',
                                                linkage = 'ward')

cust_all_var2['hierarchical_ecu'] = hierarchical_model_euc.fit_predict(cust_all_var2_scale_df)
cust_all_alive_lose['hierarchical_ecu'] = hierarchical_model_euc.fit_predict(cust_all_var2_scale_df)
print(cust_all_var2['hierarchical_ecu'].value_counts())


# In[26]:


#找出最佳分群：cosine & average
sil_score_agglo = []

for n_clusters in range(2,20):
    hierarchical_model = AgglomerativeClustering(n_clusters = n_clusters,
                                                affinity = 'cosine',
                                                linkage = 'average')
    labels = []
    labels = hierarchical_model.fit_predict(cust_all_var2_scale_df)
    sil_score_agglo.append(metrics.silhouette_score(cust_all_var2_scale_df,labels,metric = 'cosine'))
    print(sil_score_agglo)
    
sns.lineplot(list(range(2,20)),sil_score_agglo)


# In[27]:


#階層式分群(cosine)
#最後依照側影係數0.6，分為17群的作為最後分群模型的建置


# In[28]:


#建立最佳分群:cosine & average，分四群
hierarchical_model_cos = AgglomerativeClustering(n_clusters = 17,
                                                affinity = 'cosine',
                                                linkage = 'average')

#cust_all_var2:要做各群的描述性統計
#cust_all_alive_lose：要抓其他欄位使用
cust_all_var2['hierarchical_cos_7'] = hierarchical_model_cos.fit_predict(cust_all_var2_scale_df)
cust_all_alive_lose['hierarchical_cos_7'] = hierarchical_model_cos.fit_predict(cust_all_var2_scale_df)
print(cust_all_var2['hierarchical_cos_7'].value_counts())


# In[29]:


#建立gmm模型
n_components = np.arange(1,21)
models = [mixture.GaussianMixture(n,covariance_type = 'full',random_state=0).fit(cust_all_var2_scale_df) 
         for n in n_components]
plt.plot(n_components,[m.bic(cust_all_var2_scale_df) for m in models],label = 'BIC')
plt.plot(n_components,[m.aic(cust_all_var2_scale_df) for m in models],label = 'AIC')

plt.legend(loc='best')
plt.xlabel('n_components')


# In[30]:


#最後gmm模型
gmm_model = mixture.GaussianMixture(n_components = 8,
                                    covariance_type = 'full',
                                    random_state=0).fit(cust_all_var2_scale_df)

cust_all_var2['gmm_cluster'] = gmm_model.predict(cust_all_var2_scale_df)
cust_all_alive_lose['gmm_cluster'] = gmm_model.predict(cust_all_var2_scale_df)
print(cust_all_var2['gmm_cluster'].value_counts())


# In[ ]:





# In[31]:


#####模型成效比較#####


# In[32]:


#計算各分群結果的側影係數
#kmeans
kmeans_labels = kmeans_model.fit_predict(cust_all_var2_scale_df)
kmeans_sil_score = metrics.silhouette_score(cust_all_var2_scale_df,kmeans_labels,metric = 'euclidean')
kmeans_sil_score_cos = metrics.silhouette_score(cust_all_var2_scale_df,kmeans_labels,metric = 'cosine')
#cust_all_alive_lose['silhouette_score_kmeans'] = metrics.silhouette_samples(cust_all_var2_scale_df,kmeans_labels)
print('kmeans result')
print(kmeans_sil_score)
print(kmeans_sil_score_cos)

#DBSCAN
dbscan_labels = dbscan_model.fit_predict(cust_all_var2_scale_df)
db_sil_score = metrics.silhouette_score(cust_all_var2_scale_df,dbscan_labels,metric = 'euclidean')
db_sil_score_cos = metrics.silhouette_score(cust_all_var2_scale_df,dbscan_labels,metric = 'cosine')
print('dbscan result')
print(db_sil_score)
print(db_sil_score_cos)

#階層式分群-ward
ward_labels = hierarchical_model_euc.fit_predict(cust_all_var2_scale_df)
hi_sil_score_w = metrics.silhouette_score(cust_all_var2_scale_df,ward_labels,metric = 'euclidean')
hi_sil_score_cos = metrics.silhouette_score(cust_all_var2_scale_df,ward_labels,metric = 'cosine')
print('ward result')
print(hi_sil_score_w)
print(hi_sil_score_cos)

#階層式分群-cosine
cos_labels = hierarchical_model_cos.fit_predict(cust_all_var2_scale_df)
hi_sil_score_c = metrics.silhouette_score(cust_all_var2_scale_df,cos_labels,metric = 'euclidean')
hi_sil_score_c_cos = metrics.silhouette_score(cust_all_var2_scale_df,cos_labels,metric = 'cosine')
print('cos result')
print(hi_sil_score_c)
print(hi_sil_score_c_cos)

#GMM
gmm_labels = gmm_model.predict(cust_all_var2_scale_df)
gmm_sil_score = metrics.silhouette_score(cust_all_var2_scale_df,gmm_labels,metric = 'euclidean')
gmm_sil_score_cos = metrics.silhouette_score(cust_all_var2_scale_df,gmm_labels,metric = 'cosine')
print('gmm result')
print(gmm_sil_score)
print(gmm_sil_score_cos)


# In[33]:


##CHS_SCORE(越大越好)


# In[34]:


kmeans_labels = kmeans_model.fit_predict(cust_all_var2_scale_df)
kmeans_chs = metrics.calinski_harabasz_score(cust_all_var2_scale_df,kmeans_labels)
print('kmeans result')
print(kmeans_chs)

dbscan_labels = dbscan_model.fit_predict(cust_all_var2_scale_df)
dbscan_chs = metrics.calinski_harabasz_score(cust_all_var2_scale_df,dbscan_labels)
print('dbscan result')
print(dbscan_chs)

ward_labels = hierarchical_model_euc.fit_predict(cust_all_var2_scale_df)
ward_chs = metrics.calinski_harabasz_score(cust_all_var2_scale_df,ward_labels)
print('ward result')
print(ward_chs)

cos_labels = hierarchical_model_cos.fit_predict(cust_all_var2_scale_df)
cos_chs = metrics.calinski_harabasz_score(cust_all_var2_scale_df,cos_labels)
print('cos result')
print(cos_chs)

gmm_labels = gmm_model.fit_predict(cust_all_var2_scale_df)
gmm_chs = metrics.calinski_harabasz_score(cust_all_var2_scale_df,gmm_labels)
print('gmm result')
print(gmm_chs)


# In[35]:


#DAVIS(越小越好)
kmeans_labels = kmeans_model.fit_predict(cust_all_var2_scale_df)
kmeans_dbs = metrics.davies_bouldin_score(cust_all_var2_scale_df,kmeans_labels)
print('kmeans result')
print(kmeans_dbs)

dbscan_labels = dbscan_model.fit_predict(cust_all_var2_scale_df)
dbscan_dbs = metrics.davies_bouldin_score(cust_all_var2_scale_df,dbscan_labels)
print('dbscan result')
print(dbscan_dbs)

ward_labels = hierarchical_model_euc.fit_predict(cust_all_var2_scale_df)
ward_dbs = metrics.davies_bouldin_score(cust_all_var2_scale_df,ward_labels)
print('ward result')
print(ward_dbs)

cos_labels = hierarchical_model_cos.fit_predict(cust_all_var2_scale_df)
cos_dbs = metrics.davies_bouldin_score(cust_all_var2_scale_df,cos_labels)
print('cos result')
print(cos_dbs)

gmm_labels = gmm_model.fit_predict(cust_all_var2_scale_df)
gmm_dbs = metrics.davies_bouldin_score(cust_all_var2_scale_df,gmm_labels)
print('gmm result')
print(gmm_dbs)


# In[28]:


#####模型結果應用#####


# In[36]:


#kmeans建模結果 各群描述性統計
cust_all_s = cust_all_alive_lose[['ID','ID2017','ID2018','ID2019','ID2020','STATUS17_18',
                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                  'kmeans_cluster']]

kmeans_cluster_describe = cust_all_s.groupby(['kmeans_cluster','STATUS17_18']).agg(['count'],as_index = False)
print(kmeans_cluster_describe)

kmeans_cluster_describe_mean = cust_all_s.groupby(['kmeans_cluster','STATUS17_18']).agg(['mean'],as_index = False)
print(kmeans_cluster_describe_mean)

#kmeans_cluster_describe.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群各客戶狀態kmeans.csv',encoding = 'utf_8_sig')

#kmeans_cluster_describe_mean.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群各客戶狀態kmeans_mean.csv',encoding = 'utf_8_sig')


# In[30]:


#DBSCAN建模結果 各群描述性統計
cust_all_s = cust_all_alive_lose[['ID','ID2017','ID2018','ID2019','ID2020','STATUS17_18',
                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                  'dbscan_cluster']]

count_aggregation = cust_all_s.groupby(['dbscan_cluster','STATUS17_18']).agg(['count'])
print(count_aggregation)

#count_aggregation.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群各客戶狀態dbscan_model.csv',
#                         encoding = 'utf_8_sig')


# In[31]:


#階層式分群建模結果(歐基里德距離) 各群描述性統計
cust_all_s = cust_all_alive_lose[['ID','ID2017','ID2018','ID2019','ID2020','STATUS17_18',
                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                  'hierarchical_ecu']]

count_aggregation = cust_all_s.groupby(['hierarchical_ecu','STATUS17_18']).agg(['count'])
print(count_aggregation)

#count_aggregation.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群各客戶狀態hirarchical_euc.csv',
#                         encoding = 'utf_8_sig')


# In[32]:


#階層式分群建模結果(cosine距離) 各群描述性統計
cust_all_s = cust_all_alive_lose[['ID','ID2017','ID2018','ID2019','ID2020','STATUS17_18',
                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                  'hierarchical_cos_7']]


count_aggregation = cust_all_s.groupby(['hierarchical_cos_7','STATUS17_18']).agg(['count'])
print(count_aggregation)
#count_aggregation.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群各客戶狀態hierarchical_cos_7.csv'
#                         ,encoding = 'utf_8_sig')


#mean_aggregation = cust_all_s.groupby(['hierarchical_cos_7','STATUS17_18']).agg(['mean'])
#print(mean_aggregation)
#mean_aggregation.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V3\\各群個客戶狀態的變數平均.csv',encoding = 'utf_8_sig')變數平均.csv',encoding = 'utf_8_sig')


# In[33]:


#gmm分群建模結果 各群描述性統計
cust_all_s = cust_all_alive_lose[['ID','ID2017','ID2018','ID2019','ID2020','STATUS17_18',
                                  'M_MF','M_BD','M_ETF','M_SI','M_INS',
                                  'R_MF2','R_BD2','R_ETF2','R_SI2','R_INS2',
                                  'F_MF','F_BD','F_ETF','F_SI','F_INS',
                                  'gmm_cluster']]

group_aggregation = cust_all_s.groupby(['gmm_cluster']).agg(['mean'])
print(group_aggregation)


count_aggregation = cust_all_s.groupby(['gmm_cluster','STATUS17_18']).agg(['count'])
print(count_aggregation)

#count_aggregation.to_csv('C:\\Users\\137263\\Desktop\\working_data_remote\\active_loss_client_predict\\mid_term\\V2\\各群各客戶狀態gmm_cluster_8.csv',
#                         encoding = 'utf_8_sig')

mean_aggregation = cust_all_s.groupby(['gmm_cluster','STATUS17_18']).agg(['mean'])
print(mean_aggregation)


# In[ ]:


###匯出最終資料


# In[ ]:


#匯出建模結果


# In[37]:


model_store = [kmeans_model,dbscan_model,hierarchical_model_euc,hierarchical_model_cos,gmm_model]

filename = '建模中間資料\\初始模型\\cluster_model_store.pkl'

pkl.dump(model_store,open(filename,'wb'))


# In[38]:


#讀取模型pickle
filename = '建模中間資料\\初始模型\\cluster_model_store.pkl'
model_storage = pkl.load(open(filename,'rb'))

#kmeans的狀態
model_storage[0]


# In[39]:


#匯出分群標記的資料


# In[40]:


cust_all_alive_lose.head(10)


# In[41]:


cust_all_alive_lose.to_csv('建模中間資料\\初始模型\\CUST_ALL1008_CLUSTER_V3.csv',encoding = 'utf_8_sig',index = False)


# In[ ]:




