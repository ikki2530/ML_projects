import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
text = ["London Paris London", "Paris Paris London", "Paris"]
cv = CountVectorizer()

count_matrix = cv.fit_transform(text)
# print(count_matrix.toarray())


# similarities between sentences
similarity = cosine_similarity(count_matrix)

print(similarity)