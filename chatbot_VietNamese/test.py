# Test Cosine Similarity between 2 Number Lists
from numpy import dot
from numpy.linalg import norm

a = [0, 0, 1]
b = [1, 0, 0]
cos_sin= dot(a, b)/(norm(a)*norm(b))
print(cos_sin)
