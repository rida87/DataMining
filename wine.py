import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import seaborn as sns

# -------------------------
# 1. Load the dataset
# -------------------------
wine = fetch_ucirepo(id=109)  # Wine dataset
X = wine.data.features        # Feature data
y = wine.data.targets         # True labels (for evaluation only)

# Inspect the data
print("First 5 rows of features:")
print(X.head())
print("\nUnique classes in y:", y['class'].unique())

# -------------------------
# 2. Data preprocessing
# -------------------------

# Check for missing values
print("\nMissing values in X:", X.isnull().sum().sum())

# Standardize features (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""Standardization is necessary in hierarchical clustering because
 the features in the dataset have different scales
 (e.g., Proline values are much larger than Alcohol).

Hierarchical clustering relies on distance metrics (like Euclidean distance).
Without standardization, features with larger scales dominate the 
distance calculation, biasing the clustering results.
Standardizing to mean = 0 and standard deviation = 1 ensures that
 all features contribute equally to cluster formation."""
# -------------------------

# -------------------------
# 3. Hierarchical clustering
# -------------------------

# Compute the linkage matrix using Ward's method
Z = linkage(X_scaled, method='ward')


# Plot the dendrogram for the hierarchical clustering
# leaf_rotation=90 rotates leaf labels vertically for readability
# leaf_font_size=8 sets the size of leaf labels
plt.figure(figsize=(12, 6))
dendrogram(Z, labels=None, leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrogram - Hierarchical Clustering (Ward linkage)")
plt.xlabel("Wine samples")
plt.ylabel("Euclidean distance")
plt.show()

# Interpreting the plot 

"""
Identifying clusters: 

Large vertical jumps indicate where distinct clusters merge.
There appear to be 3 main clusters (orange, green, red).
These clusters merge at higher distances (~25 on the y-axis).
This matches the known structure of the Wine dataset (3 cultivars).
3️⃣ Interpreting distances
Small vertical distances at the bottom → merging of very similar wines.
Larger vertical distances at the top → merging of more dissimilar clusters.
The height of a merge indicates how different clusters are.
"""


# Choose 3 clusters (based on dendrogram)
num_clusters = 3
# criterion='maxclust' : Cut the dendrogram at the height that gives exactly N clusters
cluster_labels = fcluster(Z, num_clusters, criterion='maxclust')

# Add cluster labels to a DataFrame for easier analysis
clustered_data = X.copy()
clustered_data['Cluster'] = cluster_labels

# Compute mean and std per cluster
cluster_summary = clustered_data.groupby('Cluster').agg(['mean', 'std'])
print("\nCluster summary statistics:")
print(cluster_summary)



# 5. Evaluation using y
# -------------------------

# Flatten y to 1D array for comparison
y_true = y.values.ravel()

# Confusion matrix
cm = confusion_matrix(y_true, cluster_labels)
print("\nConfusion matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - True labels vs Clusters")
plt.xlabel("Cluster labels")
plt.ylabel("True wine class")
plt.show()