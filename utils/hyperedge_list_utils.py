from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def calculate_similarity_matrix(vecs):
    names = list(vecs)
    if not names:
        return pd.DataFrame()
    m = np.array(list(vecs.values()))
    s = cosine_similarity(m)
    np.fill_diagonal(s, -np.inf)
    return pd.DataFrame(s, index=names, columns=names)


def perform_hierarchical_grouping(session, thresh=0.8):
    vecs = session["hyperedge_avg_features"].copy()
    comp = {k: [k] for k in vecs}
    counts = {k: 1 for k in vecs}

    while len(vecs) > 1:
        sim = calculate_similarity_matrix(vecs)
        if sim.empty or sim.values.max() < thresh:
            break
        col = sim.max().idxmax()
        row = sim[col].idxmax()
        new = f"temp_{uuid.uuid4()}"
        c1, c2 = counts[row], counts[col]
        vecs[new] = (vecs[row] * c1 + vecs[col] * c2) / (c1 + c2)
        comp[new] = comp.pop(row) + comp.pop(col)
        counts[new] = c1 + c2
        vecs.pop(row), vecs.pop(col)
    return comp


def rename_groups_sequentially(raw):
    res, cnt, singles = {}, 1, []
    for k, ch in raw.items():
        if len(ch) > 1:
            res[f"Meta-Group {cnt}"] = ch
            cnt += 1
        else:
            singles.extend(ch)
    if singles:
        res["Ungrouped"] = singles
    return res


def build_row_data(groups, session):
    status = session["status_map"]
    rows = []
    for g, children in groups.items():
        for child in children:
            meta = status[child]
            rows.append(
                dict(
                    uuid=meta["uuid"],
                    name=child,
                    image_count=len(session["hyperedges"][child]),
                    status=meta["status"],
                    similarity=None,
                    group_name=g,
                )
            )
    return rows
