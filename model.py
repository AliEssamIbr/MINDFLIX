import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer

# -----------------------------
# LOAD DATA
# -----------------------------
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

data = pd.merge(ratings, movies, on="movieId")

# -----------------------------
# ENCODE USERS & MOVIES
# -----------------------------
user_ids = data['userId'].unique()
movie_ids = data['movieId'].unique()

user_map = {id: i for i, id in enumerate(user_ids)}
movie_map = {id: i for i, id in enumerate(movie_ids)}

data['user'] = data['userId'].map(user_map)
data['movie'] = data['movieId'].map(movie_map)

# -----------------------------
# GENRES (CONTENT PART)
# -----------------------------
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies['genres'])

movie_genre_map = {
    movie_id: genre_matrix[i]
    for i, movie_id in enumerate(movies['movieId'])
}

data['genres'] = data['movieId'].map(movie_genre_map)

num_users = len(user_map)
num_movies = len(movie_map)
num_genres = genre_matrix.shape[1]

# -----------------------------
# MODEL
# -----------------------------
class Recommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, 50)
        self.movie_embed = nn.Embedding(num_movies, 50)

        self.fc = nn.Sequential(
            nn.Linear(100 + num_genres, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, movie, genres):
        u = self.user_embed(user)
        m = self.movie_embed(movie)
        x = torch.cat([u, m, genres], dim=1)
        return self.fc(x).squeeze()

model = Recommender()

# -----------------------------
# TRAIN
# -----------------------------
users = torch.tensor(data['user'].values)
movies_t = torch.tensor(data['movie'].values)
genres = torch.tensor(np.stack(data['genres'].values)).float()
ratings = torch.tensor(data['rating'].values).float()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    preds = model(users, movies_t, genres)
    loss = loss_fn(preds, ratings)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")

# Save mappings (IMPORTANT)
import pickle

with open("user_map.pkl", "wb") as f:
    pickle.dump(user_map, f)

with open("movie_map.pkl", "wb") as f:
    pickle.dump(movie_map, f)

with open("movie_genre_map.pkl", "wb") as f:
    pickle.dump(movie_genre_map, f)

movies.to_csv("movies_processed.csv", index=False)
