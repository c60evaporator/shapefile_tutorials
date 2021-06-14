# %% 多重ループ
mylist = [[i, j] for i in range(10) for j in range(10) if i == j + 1]
print(mylist)

# %%
import numpy as np
d = 2                            # dimension
nb = 10000                       # database size
nq = 1000                        # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# %% sklearn
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(metric='cosine')
nn.fit(xb)
dists, result = nn.kneighbors(xq, n_neighbors=5)
print (result)
print (dists)