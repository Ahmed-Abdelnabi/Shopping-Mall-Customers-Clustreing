#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[64]:


import warnings
import numpy as np 
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage


# # Preprocessing Functions

# In[49]:


def clean_df(df):
    # Drop completely unique columns
    df.drop(["CustomerID"], axis=1, inplace=True, errors='ignore' )
    # Change data types as needed
    df['Gender'] = df['Gender'].astype('category')
    
    # Deal with outliers using IQR Method
    num_cols = df.select_dtypes('number').columns
    
    for i,col in enumerate(num_cols):
        Q1 = np.quantile(df[col], 0.25)
        Q2 = np.quantile(df[col], 0.5)
        Q3 = np.quantile(df[col], 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        lower_outliers = df[col][df[col] < lower_bound]
        upper_outliers = df[col][df[col] > upper_bound]

        df[col].replace(upper_outliers.values,upper_bound,inplace=True)
        df[col].replace(lower_outliers.values,lower_bound,inplace=True)
        
    return df

def scaling(df):
    num_cols = df.select_dtypes('number').columns
    std_scaler = StandardScaler()
    df[num_cols] = std_scaler.fit_transform(df[num_cols])
    return df,std_scaler

            
def encoding(df,encoder=None, train=True):
    if train:
        l_enc = LabelEncoder()
        df['Gender'] = l_enc.fit_transform(df['Gender'])
        return df, l_enc
    else:
        df['Gender'] = encoder.transform(df['Gender'])
        return df


# # Load Dataset

# In[95]:


df = pd.read_csv(".\Dataset\\Mall_Customers.csv")
df = clean_df(df)


# In[96]:


df.describe()


# # K-Means Clustering

# In[53]:


df_kmeans = df.copy()
df_kmeans_scaled = df.copy()
df_kmeans_scaled, scaler = scaling(df_kmeans_scaled)
df_kmeans_scaled, encoder = encoding(df_kmeans_scaled, train=True)
df_kmeans_scaled.describe()


# ### Calculating WCSS and Silhouette Score

# In[54]:


inirtia = []
silhouette_scores = []
for k in range(2, 12):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(df_kmeans_scaled)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_kmeans_scaled, labels)
    silhouette_scores.append(silhouette_avg)
    inirtia.append(kmeans.inertia_)


# In[55]:


plt.figure(figsize=(12,5))
plt.grid()
plt.plot(range(2,12), inirtia,marker='x')
plt.xlabel('number of clusters')
plt.xticks(np.linspace(2,12,11))
plt.ylabel('WCSS')
plt.title("WCSS VS Number of Clusters (Elbow Method)")
plt.show()


# **The number of clusters seems to be 6 clusters at the elbow point**

# In[56]:


plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(range(2, 12), silhouette_scores, linewidth=2, color="blue", marker="x")
plt.xlabel("Number of Clusters (K)")
plt.xticks(np.linspace(2,12,11))
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Number of Clusters")
plt.scatter(silhouette_scores.index(max(silhouette_scores))+2,max(silhouette_scores), c='r', linewidths=10 )
plt.show()


optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")


# In[57]:


kmeans = KMeans(n_clusters=6,random_state=42)
kmeans.fit(df_kmeans_scaled)


# In[58]:


df_kmeans['clusters'] = kmeans.predict(df_kmeans_scaled)
df_kmeans


# In[59]:


fig = plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
sns.scatterplot(data=df_kmeans, x='Annual Income (k$)', y='Spending Score (1-100)',hue='clusters', palette='viridis')
plt.subplot(2,2,2)
sns.scatterplot(data=df_kmeans, x='Annual Income (k$)', y='Age', hue='clusters', palette='viridis')
plt.subplot(2,2,3)
sns.scatterplot(data=df_kmeans, x='Age', y='Spending Score (1-100)', hue='clusters', palette='viridis')

plt.show()


# - Cluster 0: Wise Moderates (Mid Earner - Mid Spender, Age above 40) - Boomer Moderates
# - Cluster 1: Elite Extravagants (High Earner - High Spender) - Luxury Lifestylers
# - Cluster 2: Young Balancers (Mid Earner - Mid Spender, Age below 40) - Gen Z Balancers
# - Cluster 3: Frugal Simplicity (Low Earner - Low Spender) - Struggling Savers
# - Cluster 4: Wealthy Savers (High Earner - Low Spender) - Wealthy Savers
# - Cluster 5: Reckless Youngsters (Low Earner - High Spender, Age below 40) - Gen Z Broke and Bold

# In[99]:


clusters_names_kmean = {
    0: 'Boomer Moderates: A wise elder, you cherish balance and enjoy the calm of life.',
    1: 'Luxury Lifestylers: A refined connoisseur, you seek elegance and embrace the finer things.',
    2: "Gen Z Balancers: A mindful earner, you manage a moderate income and spend with balance.",
    3: 'Struggling Savers: A resilient planner, you work hard and stretch every dollar.',
    4: 'Wealthy Savers: A cautious investor, you enjoy your wealth but plan wisely for the future.',
    5: 'Gen Z Broke and Bold: A daring go-getter, you live boldly today with little resources without worrying about tomorrow.'
}


# # Agglomerative Hierarchical Clustering

# In[63]:


df_agglomerative = df.copy()
df_agglomerative_scaled = df.copy()
df_agglomerative_scaled, scaler = scaling(df_agglomerative_scaled)
df_agglomerative_scaled, encoder = encoding(df_agglomerative_scaled, train=True)
df_agglomerative_scaled.describe()


# In[65]:


linkage_data = linkage(df_agglomerative_scaled, method='ward', metric='euclidean')
fig = plt.figure(figsize=(15,8))
dendrogram(linkage_data)
plt.axhline(y=7, c='r')
plt.show()


# **A good cut off point is drawn as the red horizontal line which indicate 6 clusters**

# **Let's use Silhoutte Score method to confirm**

# In[66]:


sil_agg = []
for i in range(2,15):
    agg_hierarchical = AgglomerativeClustering(n_clusters=i)
    agg_hierarchical.fit(df_agglomerative_scaled)
    sil_agg.append(silhouette_score(df_agglomerative_scaled,agg_hierarchical.labels_))

plt.figure(figsize=(9,5))
plt.plot(range(2,15),sil_agg)
plt.scatter(sil_agg.index(max(sil_agg))+2,max(sil_agg), c='r', linewidths=10 )
plt.scatter(sil_agg.index(max(sil_agg))+3,max(sil_agg), c='r', linewidths=10 )
plt.title("Silhoutte Score for Agglomerative Hierarchical Model")
plt.xticks(np.linspace(2,14,13))
plt.grid()
plt.show()


# **It shows that either 5 or 6 clusters are fine, we will stick with 6 clusters to be coherent with Kmeans Model**

# In[67]:


agg_hierarchical = AgglomerativeClustering(n_clusters=6)
agg_hierarchical.fit(df_agglomerative_scaled)


# In[68]:


fig = plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                hue = agg_hierarchical.labels_ , palette='viridis' )

plt.subplot(2,2,2)
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Age'],
                hue = agg_hierarchical.labels_ , palette='viridis' )

plt.subplot(2,2,3)
sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'],
                hue = agg_hierarchical.labels_ , palette='viridis' )


plt.show()


# In[100]:


clusters_names_agg = {
    0:'Boomer Moderates: A wise elder, you cherish balance and enjoy the calm of life.',
    1: 'Struggling Savers: A resilient planner, you work hard and stretch every dollar.',
    2: 'Wealthy Savers: A cautious investor, you enjoy your wealth but plan wisely for the future.',
    3: 'Luxury Lifestylers: A refined connoisseur, you seek elegance and embrace the finer things.',
    4: "Gen Z Balancers: A mindful earner, you manage a moderate income and spend with balance.",
    5: 'Gen Z Broke and Bold: A daring go-getter, you live boldly today with little resources without worrying about tomorrow.'  
}


# # DBSCAN Clustering

# In[101]:


df_dbscan = df.copy()
df_dbscan_scaled, scaler = scaling(df_dbscan)
df_dbscan_scaled, encoder = encoding(df_dbscan, train=True)
df_dbscan_scaled.describe()


# In[235]:


# Let's use the Silhoutte Score method to confirm
param = {}
min_sample = []
epsilon = []
hyp_param = []
sil_dbscan = []
for i in range(1,20):
    for j in np.linspace(0.2,2,50):
        try:
            dbscan = DBSCAN(eps=j,min_samples=i)
            dbscan.fit(df_dbscan_scaled)
            sil = silhouette_score(df_dbscan_scaled,dbscan.labels_)
        except ValueError:
            sil_dbscan.append(0)
        else:
            sil_dbscan.append(sil)
            
        param[sil] = [i,j]
        min_sample.append(i)
        epsilon.append(j)
        hyp_param.append((j,i))

plt.figure(figsize=(18,5))
plt.plot(range(len(hyp_param)),sil_dbscan)
plt.title("Silhoutte Score for DBSCAN Model")
plt.xlabel("min_samples")
plt.xticks(np.linspace(0,len(hyp_param),(len(hyp_param)//20)+1), rotation=90)
# plt.
plt.grid()
plt.show()


# **The suspected indices are: (70, 85, 131, 100, 115, 54)**

# In[248]:


max(sil_dbscan)


# In[249]:


param[0.3019572324123013]


# In[250]:


dbscan = DBSCAN(eps=1.1551020408163266,min_samples=6)
dbscan.fit(df_dbscan_scaled)


# In[254]:


fig = plt.figure(figsize=(15,12))
plt.subplot(2,2,1)
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                hue = dbscan.labels_ , palette='viridis' )

plt.subplot(2,2,2)
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Age'],
                hue = dbscan.labels_ , palette='viridis' )

plt.subplot(2,2,3)
sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'],
                hue = dbscan.labels_ , palette='viridis' )

plt.show()


# **It appears that DBSCAN is not suitable for this dataset because it is not dense enough, we found that at the best Silhouette score it clusters the data as one cluster only, and no matter how much we tried to tune the hyperparameters it doesn't get a better result**

#  

# # Pickling the Models, the Scaler and the Encoder

# In[76]:


df['kmeans labels'] = kmeans.labels_
df['agglomerative labels'] = agg_hierarchical.labels_
df['kmeans labels'] = df['kmeans labels'].map(clusters_names_kmean)
df['agglomerative labels'] = df['agglomerative labels'].map(clusters_names_agg)
df.to_csv("clustering_solved.csv", index=False)


# In[102]:


import pickle 

with open('kmeans.pk','wb') as file:
    pickle.dump(kmeans, file)
    
with open('agg.pk','wb') as file:
    pickle.dump(agg_hierarchical, file)
    
with open('scaler.pk','wb') as file:
    pickle.dump(scaler, file)
    
with open('encoder.pk','wb') as file:
    pickle.dump(encoder, file)

with open('clusters_kmean.pk', 'wb') as file:
    pickle.dump(clusters_names_kmean, file)
    
with open('clusters_agglomerative.pk', 'wb') as file:
    pickle.dump(clusters_names_agg, file)    

