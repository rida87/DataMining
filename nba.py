# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Part 1: Data Exploration
print("=" * 50)
print("PART 1: DATA EXPLORATION")
print("=" * 50)

# 1. Load the dataset
print("1. Loading dataset...")
df = pd.read_csv('nba_2013.csv')  # Make sure the file is in your working directory

# 2. Examine the first few rows
print("\n2. First few rows of the dataset:")
print(df.head())

# Basic dataset info
print("\nDataset shape:", df.shape)
print("\nDataset info:")
print(df.info())

# 3. Check for missing values
print("\n3. Missing values:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Handle missing values - fill numeric columns with median
print("\nHandling missing values...")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# For categorical columns, fill with mode
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after handling:", df.isnull().sum().sum())

# 4. Explore distributions of key statistics
print("\n4. Exploring distributions of key statistics...")

# Select key statistics to visualize
key_stats = ['pts', 'ast', 'trb', 'mp']

# Create histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, stat in enumerate(key_stats):
    axes[i].hist(df[stat].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {stat.upper()}')
    axes[i].set_xlabel(stat.upper())
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Distributions of Key Basketball Statistics', y=1.02, fontsize=16)
plt.show()

# Create boxplots
plt.figure(figsize=(12, 6))
df[key_stats].boxplot()
plt.title('Boxplots of Key Basketball Statistics')
plt.xticks(rotation=45)
plt.show()

# Part 2: k-NN Classification
print("\n" + "=" * 50)
print("PART 2: k-NN CLASSIFICATION")
print("=" * 50)

# 1. Select numeric features for classification
print("1. Selecting features for position prediction...")
features_classification = [
    'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 
    'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf', 'pts'
]

# Create feature matrix X and target vector y
X_class = df[features_classification]
y_class = df['pos']

print(f"Features shape: {X_class.shape}")
print(f"Target classes: {y_class.unique()}")
print(f"Class distribution:\n{y_class.value_counts()}")

# 2. Split the dataset
print("\n2. Splitting dataset into training and testing sets...")
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

print(f"Testing set size: {X_test_class.shape[0]}")

# 3. Scale the numeric features
print("\n3. Scaling features...")
scaler_class = StandardScaler()
X_train_class_scaled = scaler_class.fit_transform(X_train_class)
X_test_class_scaled = scaler_class.transform(X_test_class)

# 4. Train k-NN classifier
print("\n4. Training k-NN classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_class_scaled, y_train_class)

# 5. Evaluate the model
print("\n5. Evaluating the model...")
y_pred_class = knn_classifier.predict(X_test_class_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"Accuracy with k=5: {accuracy:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_class, y_pred_class)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=knn_classifier.classes_, 
            yticklabels=knn_classifier.classes_)
plt.title('Confusion Matrix - Position Prediction (k=5)')
plt.xlabel('Predicted Position')
plt.ylabel('Actual Position')
plt.show()



### Confusion Matrix Interpretaion

#Confusion Matrix:
#[[13  5  1  2  2]
# [ 3  5  0  4  1]
 #[ 0  0 10  1  7]
 #[ 1  5  1  8  6]
# [ 2  1  5  9  5]]
#   C  F  G  PG  PG

"""
The model predicted:

13 players as C → Correct prediction

5 players as PF → Wrong

1 player as PG → Wrong

2 players as SF → Wrong

2 players as SG → Wrong
"""


# Classification report
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))

# 6. Experiment with different k values
print("\n6. Experimenting with different k values...")
k_values = [1, 3, 5, 7, 10, 15]
accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_class_scaled, y_train_class)
    y_pred_temp = knn_temp.predict(X_test_class_scaled)
    acc = accuracy_score(y_test_class, y_pred_temp)
    accuracies.append(acc)
    print(f"k={k}: Accuracy = {acc:.4f}")

# Plot accuracy vs k
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('k-NN Classification: Accuracy vs k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Part 3: k-NN Regression
print("\n" + "=" * 50)
print("PART 3: k-NN REGRESSION")
print("=" * 50)

# 1. Select predictor features (exclude pts and pos)
print("1. Selecting features for points prediction...")
features_regression = [col for col in features_classification if col not in ['pts']]
X_reg = df[features_regression]
y_reg = df['pts']

print(f"Regression features shape: {X_reg.shape}")

# 2. Split the dataset
print("\n2. Splitting dataset for regression...")
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train_reg.shape[0]}")
print(f"Testing set size: {X_test_reg.shape[0]}")

# 3. Scale the numeric features
print("\n3. Scaling features for regression...")
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

# 4. Train k-NN regressor
print("\n4. Training k-NN regressor...")
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_reg_scaled, y_train_reg)

# 5. Evaluate the model
print("\n5. Evaluating regression model...")
y_pred_reg = knn_regressor.predict(X_test_reg_scaled)

# Calculate metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error (k=5): {mse:.4f}")
print(f"R² Score (k=5): {r2:.4f}")

# 6. Experiment with different k values
print("\n6. Experimenting with different k values for regression...")
k_values_reg = [1, 3, 5, 7, 10, 15, 20]
mse_scores = []
r2_scores = []

for k in k_values_reg:
    knn_temp = KNeighborsRegressor(n_neighbors=k)
    knn_temp.fit(X_train_reg_scaled, y_train_reg)
    y_pred_temp = knn_temp.predict(X_test_reg_scaled)
    
    mse_temp = mean_squared_error(y_test_reg, y_pred_temp)
    r2_temp = r2_score(y_test_reg, y_pred_temp)
    
    mse_scores.append(mse_temp)
    r2_scores.append(r2_temp)
    
    print(f"k={k}: MSE = {mse_temp:.4f}, R² = {r2_temp:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(k_values_reg, mse_scores, marker='o', linestyle='-', color='red')
ax1.set_title('k-NN Regression: MSE vs k Value')
ax1.set_xlabel('k Value')
ax1.set_ylabel('Mean Squared Error')
ax1.grid(True)

ax2.plot(k_values_reg, r2_scores, marker='o', linestyle='-', color='green')
ax2.set_title('k-NN Regression: R² Score vs k Value')
ax2.set_xlabel('k Value')
ax2.set_ylabel('R² Score')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Part 4: Player Similarity Exploration
print("\n" + "=" * 50)
print("PART 4: PLAYER SIMILARITY EXPLORATION")
print("=" * 50)

# 1. Pick a player (using LeBron James as example)
print("1. Selecting a player for similarity analysis...")
# Find LeBron James in the dataset
target_player_name = "LeBron James"
target_player = df[df['player'] == target_player_name]

if len(target_player) == 0:
    # If LeBron not found, pick the first player
    target_player = df.iloc[0:1]
    target_player_name = target_player['player'].values[0]
    print(f"LeBron James not found. Using {target_player_name} instead.")
else:
    print(f"Found {target_player_name}")

print(f"\nTarget Player: {target_player_name}")
print(f"Position: {target_player['pos'].values[0]}")
print(f"Points: {target_player['pts'].values[0]}")
print(f"Assists: {target_player['ast'].values[0]}")
print(f"Rebounds: {target_player['trb'].values[0]}")

# 2. Use k-NN to find 5 most similar players
print("\n2. Finding 5 most similar players...")

# Use the same features as in classification but exclude the target player
similarity_features = features_classification

# Prepare the data
X_similarity = df[similarity_features]
player_names = df['player']
player_positions = df['pos']

# Scale the features
scaler_similarity = StandardScaler()
X_similarity_scaled = scaler_similarity.fit_transform(X_similarity)

# Find the index of our target player
target_idx = df[df['player'] == target_player_name].index[0]

# Use k-NN to find nearest neighbors
knn_similarity = KNeighborsRegressor(n_neighbors=6)  # 6 because it will include the player itself
knn_similarity.fit(X_similarity_scaled, range(len(X_similarity_scaled)))

# Find distances and indices of nearest neighbors
distances, indices = knn_similarity.kneighbors([X_similarity_scaled[target_idx]])

# Exclude the player itself (first in the list)
similar_indices = indices[0][1:]
similar_distances = distances[0][1:]

print(f"\n5 most similar players to {target_player_name}:")
print("-" * 50)

# Display similar players
similar_players = []
for i, (idx, dist) in enumerate(zip(similar_indices, similar_distances)):
    player_name = player_names.iloc[idx]
    position = player_positions.iloc[idx]
    points = df.iloc[idx]['pts']
    assists = df.iloc[idx]['ast']
    rebounds = df.iloc[idx]['trb']
    
    similar_players.append({
        'name': player_name,
        'position': position,
        'points': points,
        'assists': assists,
        'rebounds': rebounds,
        'similarity_distance': dist
    })
    
    print(f"{i+1}. {player_name} (Pos: {position})")
    print(f"   Points: {points}, Assists: {assists}, Rebounds: {rebounds}")
    print(f"   Similarity Distance: {dist:.4f}")
    print()

# 3. Compare stats and positions
print("3. Statistical comparison:")
print(f"Target player position: {target_player['pos'].values[0]}")
print("Similar players positions:", [p['position'] for p in similar_players])

# 4. Visualize neighbors
print("\n4. Creating visualization...")

# Prepare data for scatter plot
all_players_data = [{
    'name': target_player_name,
    'position': target_player['pos'].values[0],
    'points': target_player['pts'].values[0],
    'assists': target_player['ast'].values[0],
    'type': 'Target'
}] + [{
    'name': p['name'],
    'position': p['position'],
    'points': p['points'],
    'assists': p['assists'],
    'type': 'Similar'
} for p in similar_players]

viz_df = pd.DataFrame(all_players_data)

# Create scatter plot
plt.figure(figsize=(12, 8))
colors = {'Target': 'red', 'Similar': 'blue'}

for player_type in ['Target', 'Similar']:
    subset = viz_df[viz_df['type'] == player_type]
    plt.scatter(subset['points'], subset['assists'], 
                c=colors[player_type], label=player_type, s=100, alpha=0.7)
    
    # Add player names as annotations
    for i, row in subset.iterrows():
        plt.annotate(row['name'], (row['points'], row['assists']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Points per Game (PTS)')
plt.ylabel('Assists per Game (AST)')
plt.title(f'Player Similarity: {target_player_name} and 5 Most Similar Players')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Additional analysis: Compare key statistics
print("\n" + "=" * 50)
print("DISCUSSION AND ANSWERS TO QUESTIONS")
print("=" * 50)

print("\nClassification Questions:")
print("• Which positions are easiest to predict? Which are hardest?")
print("  - Typically, centers (C) and point guards (PG) are easier to predict")
print("  - due to their distinctive statistical profiles.")
print("  - Combo positions like SF/PF might be harder to distinguish.")

print("\n• How do shooting stats contribute to predicting position?")
print("  - 3-point percentage (x3p.) helps identify guards and stretch forwards")
print("  - 2-point percentage (x2p.) and rebounds help identify big men")
print("  - Free throw percentage (ft.) often correlates with guard positions")

print("\nRegression Questions:")
print("• Which features seem most important for predicting points?")
print("  - Field goals attempted (fga), minutes played (mp), and field goal percentage (fg.)")
print("  - are typically strong predictors of scoring output.")

print("\n• Does increasing k improve or worsen predictions?")
print("  - Initially, increasing k can improve predictions by reducing overfitting")
print("  - However, too large k can underfit the model")
print("  - Optimal k is typically found through experimentation")

print("\nSimilarity Analysis:")
print("• Do the nearest neighbors make sense?")
print("  - Similar players should have comparable statistical profiles")
print("  - Players in similar positions with similar roles should cluster together")
print("  - The visualization helps validate the similarity relationships")"""
