from LSH import LSH
from sklearn.datasets import make_blobs
import numpy as np

data, labels = make_blobs(n_samples=100, centers=10, random_state=42)

# Create LSH index
lsh_index = LSH(data, hash_size=10)

# Query for a random point
query_point = np.random.randn(data.shape[1])
result = lsh_index.query(query_point, num_candidates=5)

print("Query Point:", query_point)
print("Nearest Neighbors:", result)

for i in result:
    print(labels[i[0]])