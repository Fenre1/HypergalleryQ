import numpy as np
from typing import Tuple, Union
from fcmeans import FCM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fuzzy_cmeans_cluster(
    features: Union[np.ndarray, "np.ndarray"],
    out_dim: int,
    threshold: float,
    m: float = 1.1,
) -> Tuple[np.ndarray, FCM]:
    """Cluster features using Fuzzy C-Means and generate a hypergraph."""
    X = np.array(features, dtype=np.float32)
    X_centered = StandardScaler(with_std=False).fit_transform(X)
    pca = PCA(n_components=256, svd_solver="randomized", random_state=0)
    X_reduced = pca.fit_transform(X_centered.astype(np.float32))
    fcm = FCM(n_clusters=out_dim, m=m)
    fcm.fit(X_reduced)
    membership = fcm.u
    hypergraph = (membership > threshold).astype(int)
    return hypergraph, fcm

__all__ = ["fuzzy_cmeans_cluster"]