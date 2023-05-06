# TensorFlow-Tuturial
This tutorial covers how to create a basic rating system from a "dummy" dataset found on TensorFlow

Software Documentation
Introduction
This software is a movie recommendation system built using TensorFlow, TensorFlow Datasets, and TensorFlow Recommenders (TFRS) libraries. It uses the MovieLens dataset, which contains information about movie ratings and features, to train a model that recommends movies to users based on their preferences.

Libraries
The software uses the following libraries:

TensorFlow: an open-source machine learning library.
TensorFlow Datasets: a library of ready-to-use datasets for TensorFlow.
TensorFlow Recommenders (TFRS): a library for building recommender systems with TensorFlow.
Data
The software loads the MovieLens dataset, which contains two splits: "train" and "test". The "train" split is used to train the model, and the "test" split is used to evaluate its performance. The dataset contains information about movie ratings and features, such as movie titles and user IDs.

Models
The software defines a TFRS model called MovieLensModel. This model uses two sub-models, one for user embeddings and one for movie embeddings, which are used to represent users and movies as vectors in a high-dimensional space. The MovieLensModel also defines a retrieval task, which is used to retrieve the top-k most relevant movies for a given user.

Workflow
The software follows the following workflow:

Load the MovieLens dataset.
Define user and movie vocabularies to convert user IDs and movie titles into integer indices for embedding layers.
Define the MovieLensModel using user and movie sub-models and a retrieval task.
Compile the model using the Adagrad optimizer.
Train the model using the "train" split of the dataset.
Index the movie embeddings using a brute-force indexing approach.
Get recommendations for a given user by querying the index with the user embeddings.
Example
The following is an example of how to use the software to get movie recommendations for a given user:

python
Copy code
# Load the MovieLens dataset.
ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x: x["movie_title"])

# Define user and movie vocabularies.
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)

# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define the retrieval task.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
))

# Define the MovieLensModel.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train the model.
model.fit(ratings.batch(4096), epochs=3)

# Index the movie embeddings.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(
    movies.batch(100).map(lambda title: (title, model.movie_model(title)))
)

