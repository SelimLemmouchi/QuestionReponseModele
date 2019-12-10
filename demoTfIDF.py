import pandas as pd
data = pd.read_csv("jokes.csv")
# print(data.columns)
# print(data.describe())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
vectorizer = TfidfVectorizer()
vectorizer.fit(data.values.ravel())

# Read a question from the user
question = [input('Entrez votre question : \n')]
question = vectorizer.transform(question)

# Rank all the questions using cosine similarity to the input question
rank = cosine_similarity(question, vectorizer
                         .transform(data['Question'].values))

# Grab the response
top = np.argsort(rank, axis=-1).T[-1:].tolist()

# Print the response
for item in top:
    print(data['Answer'].iloc[item].values[0])