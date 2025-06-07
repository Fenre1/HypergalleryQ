from functools import partial
from sklearn.metrics.pairwise import cosine_similarity as _cos

# Expose a single callable; swap it out later if you like.
SIM_METRIC = partial(_cos, dense_output=True)      #  (A, B) -> ndarray