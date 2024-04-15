#!/usr/bin/env python
# coding: utf-8

# # Task 2

# # Create a K-means clustering algorithm to group customers of retail store based on their purchase history.

# In[21]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings


# In[22]:


# To ignore specific warnings:
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto'")


# In[23]:


# Alternatively, to ignore all FutureWarnings:
warnings.filterwarnings("ignore", category=FutureWarning)


# In[24]:


# To ignore UserWarning:
warnings.filterwarnings("ignore", category=UserWarning)


# In[25]:


import warnings

# Specific warnings ko ignore karne ke liye:
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto'")

# Ya fir, saare FutureWarning ko ignore karne ke liye:
warnings.filterwarnings("ignore", category=FutureWarning)

# UserWarning ko ignore karna:
warnings.filterwarnings("ignore", category=UserWarning)


# In[26]:


# Example purchase history data (replace this with your actual data)
purchase_history = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Grocery': [100, 150, 300, 50, 250],
    'Clothing': [200, 250, 400, 100, 350],
    'Electronics': [300, 350, 500, 200, 450]
}


# In[27]:


# Convert data to DataFrame
df = pd.DataFrame(purchase_history)


# In[28]:


# Extract features (excluding CustomerID)
features = df.drop('CustomerID', axis=1)


# In[29]:


# Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)


# In[30]:


# Define the number of clusters
num_clusters = 3


# In[31]:


# Apply K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(scaled_features)


# In[32]:


# Get cluster labels
df['Cluster'] = kmeans.labels_


# In[33]:


# Print cluster centers
print("Cluster centers:\n", scaler.inverse_transform(kmeans.cluster_centers_))


# In[34]:


# Visualize the clusters
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(num_clusters):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Grocery'], cluster_data['Clothing'], color=colors[i], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='*', label='Centroids')
plt.xlabel('Grocery')
plt.ylabel('Clothing')
plt.title('Customer Segmentation based on Purchase History')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




