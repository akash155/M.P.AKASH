# Complete Guide to Build a Personalized Recommendation System

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Step 2: Load and Explore Data
data = pd.read_csv('genome_scores.csv')  # Replace with your dataset
print(data.head())

# Step 3: Data Preprocessing
data.dropna(inplace=True)
data['user_id'] = data['movieId'].astype(int)
data['item_id'] = data['tagId'].astype(int)
data['rating'] = data['relevance'].astype(float)

# Step 4: Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 5: Create User-Item Interaction Matrix
interaction_matrix = train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Step 6: Calculate Similarity Matrix
user_similarity = cosine_similarity(interaction_matrix)
item_similarity = cosine_similarity(interaction_matrix.T)

# Step 7: Create Recommendation Function
def recommend(user_id, interaction_matrix, similarity_matrix, top_n=5):
    user_index = interaction_matrix.index.get_loc(user_id)
    similarity_scores = similarity_matrix[user_index]
    
    scores = interaction_matrix.values.T @ similarity_scores
    scores_df = pd.DataFrame({
        'item_id': interaction_matrix.columns,
        'score': scores
    })
    
    recommended_items = scores_df.sort_values(by='score', ascending=False).head(top_n)
    return recommended_items

# Step 8: Test Recommendation
sample_user = interaction_matrix.index[0]  # Replace with a valid user ID
recommendations = recommend(sample_user, interaction_matrix, user_similarity)
print(recommendations)

# Step 9: Evaluate Model
def evaluate_model(test_data, interaction_matrix, similarity_matrix):
    predictions = []
    for index, row in test_data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        actual = row['rating']

        if user_id in interaction_matrix.index and item_id in interaction_matrix.columns:
            predicted_scores = recommend(user_id, interaction_matrix, similarity_matrix, top_n=len(interaction_matrix.columns))
            predicted_rating = predicted_scores.loc[predicted_scores['item_id'] == item_id, 'score']
            predicted_rating = predicted_rating.iloc[0] if not predicted_rating.empty else 0
        else:
            predicted_rating = np.nan
        
        predictions.append({'user_id': user_id, 'item_id': item_id, 'actual': actual, 'predicted': predicted_rating})

    predictions_df = pd.DataFrame(predictions).dropna()
    mse = mean_squared_error(predictions_df['actual'], predictions_df['predicted'])
    return mse

mse = evaluate_model(test_data, interaction_matrix, user_similarity)
print(f'Mean Squared Error: {mse}')

# Step 10: Save Code to GitHub
# - Initialize a GitHub repository and push this code file to the repository.
# - Use the following commands in your terminal:

# git init
# git add personalized_recommendation_system.py
# git commit -m "Add personalized recommendation system project"
# git branch -M main
# git remote add origin <your-repository-url>
# git push -u origin main
