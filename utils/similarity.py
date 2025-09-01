from functools import partial
from sklearn.metrics.pairwise import cosine_similarity as _cos

SIM_METRIC = partial(_cos, dense_output=True)      