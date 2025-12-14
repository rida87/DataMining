import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the Dataset
data = pd.read_csv("Mall_Customers.csv")
print(data.head())

# Check main features
print('Columns:',data.columns)


# Select Features (Annual Income & Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Create a StandardScaler object
scaler = StandardScaler()

#converts each feature to have mean = 0 and standard deviation = 1
X_scaled = scaler.fit_transform(X)


#Why are Annual Income and Spending Score commonly used for segmentation?
#Because income and spending behavior are highly relevant to marketing decisions. Customers with similar income and spending score often behave similarly,
#making them easy to group into meaningful segments (e.g., high income–low spending).



# Create scatter plot
plt.figure(figsize=(8,6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
            c='blue', s=50, alpha=0.6)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)
plt.show()

#Do you observe any natural grouping in the scatter plot?
#Natural groups are :
#High income, high spending: Rich people who shop a lot.
#High income, low spending: Rich people who don’t spend much.
#Low income, high spending: Not-so-rich people who spend a lot.
#Low income, low spending: Not-so-rich people who spend little.
#Medium income, medium spending: People with average money and average spending.

# Determine Optimal Number of Clusters using Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    #kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# EXPLANATION OF : init='k-means++'
#init='k-means++' is a smarter way to choose the starting centroids:
#It selects the first centroid randomly.#For the next centroids, it prefers points that are far away from the already chosen centroids.
#This reduces the chances of poor clustering and helps the algorithm converge faster and more reliably.

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.show()

#The elbow method usually shows a clear bend at k = 5. The bend is the turning point where the improvement slows down.
#It’s considered the optimal number of clusters. 
#This means that adding more clusters beyond 5 does not significantly reduce the within-cluster variance, so k = 5 is the most efficient number of clusters.


# Apply K-Means (Assume k=5 from Elbow Method)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster labels to original data
data['Cluster'] = clusters
print('Data Clusters = ', data['Cluster'].value_counts())


# Visualize the Clusters
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X[clusters==i, 0], X[clusters==i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], s=200, c='yellow', marker='X', label='Centroids')
plt.title("K-Means Clustering (Mall Customers)")
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.show()

# Interpretation
#Cluster 1: Give special VIP deals.
#Cluster 2: Give discounts to make them spend more.
#Cluster 3: Give normal promotions.
#Cluster 4: Give rewards to keep them coming back.
#Cluster 5: Give cheap deals for low spenders.


