"""Local final step: stacking of the out-of-fold probabilities (5 folds trained on Colab) + catalogue neighbours.

  python stack_neighbours.py [image tfidf bert_e3_l256]
Reads oof/oof5/<model>_f<k>.npz (copied from Drive/<DRIVE_DIR>/out/oof5/), with the same folds as
colab/rakuten_colours_5fold.ipynb (KFold 5, seed 42). Compares the stacker alone (logistic regression) with the
stacker + neighbours (one LightGBM per colour) and writes the submission of the better one.
"""
import sys
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from common import DATA, OOF, SUB, binarize, fit_thresholds, load, wf1, write_submission

names = sys.argv[1:] or ["image", "tfidf", "bert_e3_l256"]
OOF5 = OOF / "oof5"
tr, te, Y, colors = load()
Yi = Y.astype(int)
FOLDS = list(KFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr))))   # same as on Colab
half = np.random.default_rng(1).permutation(len(Y)) < len(Y) // 2


def honest(p):
    """Thresholds tuned on one half of the training set (out-of-fold probabilities), F1 measured on the other half."""
    return wf1(Yi[~half], binarize(p[~half], fit_thresholds(Yi[half], p[half])))


def merge(name):
    oof, test = np.zeros(Y.shape, np.float32), np.zeros((len(te), Y.shape[1]), np.float32)
    for k, (_, b) in enumerate(FOLDS):
        z = np.load(OOF5 / f"{name}_f{k}.npz")
        oof[b] = z["val"]
        test += z["test"] / len(FOLDS)
    return oof, test


P = {n: merge(n) for n in names}
for n, (o, _) in P.items():
    print(f"{n:15s} alone: {honest(o):.4f}", flush=True)

logit = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
S_tr = np.hstack([logit(o) for o, _ in P.values()])
S_te = np.hstack([logit(t) for _, t in P.values()])
META = list(KFold(5, shuffle=True, random_state=7).split(S_tr))   # meta-model folds, different from FOLDS

# 1) stacker alone: one logistic regression per colour
lr = lambda: LogisticRegression(C=0.5, max_iter=2000)
lr_oof = np.zeros(Y.shape)
for a, b in META:
    for j in range(len(colors)):
        lr_oof[b, j] = lr().fit(S_tr[a], Yi[a, j]).predict_proba(S_tr[b])[:, 1]
lr_te = np.column_stack([lr().fit(S_tr, Yi[:, j]).predict_proba(S_te)[:, 1] for j in range(len(colors))])
s_lr = honest(lr_oof)
print(f"stacking without neighbours: {s_lr:.4f}", flush=True)


# 2) neighbours: image_file_name = <shop>_<item>_1.jpg; a shop lists its products in batches
def ids(df):
    p = df.image_file_name.str.extract(r"^(\d+)_(\d+)")
    shop, item = p[0].astype(np.int64).values, p[1].astype(np.int64).values
    return shop, shop * 10**9 + item


shop_tr, key_tr = ids(tr)
shop_te, key_te = ids(te)
order = np.argsort(key_tr, kind="stable")
s_key, s_shop, s_Y = key_tr[order], shop_tr[order], Y[order]
rank = np.empty(len(order), dtype=np.int64)
rank[order] = np.arange(len(order))
K = 5


def neigh(qkey, qshop, self_rank=None):
    """Mean colours of the K training neighbours on each side (same shop), the product itself excluded."""
    pos = np.searchsorted(s_key, qkey)
    acc, w = np.zeros((len(qkey), len(colors)), np.float32), np.zeros(len(qkey), np.float32)
    near = np.full(len(qkey), 99.0, np.float32)
    for side in (-1, 1):
        for j in range(1, K + 1):
            if self_rank is not None:
                idx = self_rank + side * j
            else:
                idx = pos - j if side < 0 else pos + j - 1
            ok = (idx >= 0) & (idx < len(s_key))
            idx = idx.clip(0, len(s_key) - 1)
            ok &= s_shop[idx] == qshop
            acc += ok[:, None] * s_Y[idx]
            w += ok
            if j == 1:
                near = np.minimum(near, np.where(ok, np.log1p(np.abs(s_key[idx] - qkey)), 99.0))
    return np.hstack([acc / np.maximum(w, 1)[:, None], w[:, None], near[:, None]])


# the shop's usual colours, out-of-fold (removing only the product itself would leak its colour)
M, prior = 5.0, Y.mean(0)


def shop_stats(rows):
    g = pd.DataFrame(Y[rows], columns=colors).assign(shop=shop_tr[rows]).groupby("shop")
    return g[colors].sum(), g.size()


def shop_prior(shop, sums, cnt):
    sm = sums.reindex(shop).fillna(0).values
    n = cnt.reindex(shop).fillna(0).values
    return np.hstack([(sm + M * prior) / (n[:, None] + M), np.log1p(n)[:, None]]).astype(np.float32)


V_tr = np.zeros((len(tr), len(colors) + 1), np.float32)
for a, b in FOLDS:
    V_tr[b] = shop_prior(shop_tr[b], *shop_stats(a))
V_te = shop_prior(shop_te, *shop_stats(np.arange(len(tr))))
F_tr = np.hstack([S_tr, neigh(key_tr, shop_tr, rank), V_tr]).astype(np.float32)
F_te = np.hstack([S_te, neigh(key_te, shop_te), V_te]).astype(np.float32)
print("features:", F_tr.shape, flush=True)

params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=30, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, n_jobs=10)
gb_oof = np.zeros(Y.shape)
for i, (a, b) in enumerate(META):
    for j in range(len(colors)):
        gb_oof[b, j] = lgb.LGBMClassifier(**params).fit(F_tr[a], Yi[a, j]).predict_proba(F_tr[b])[:, 1]
    print("meta fold", i, "done", flush=True)
s_gb = honest(gb_oof)
print(f"stacking + neighbours (LightGBM): {s_gb:.4f}   |   stacking without neighbours: {s_lr:.4f}", flush=True)

# 3) submission: the better of the two, thresholds tuned on the 212,120 out-of-fold rows
if s_gb > s_lr:
    gb_te = np.column_stack([lgb.LGBMClassifier(**params).fit(F_tr, Yi[:, j]).predict_proba(F_te)[:, 1]
                             for j in range(len(colors))])
    fusion, score, p_oof, p_te = "neighbours", s_gb, gb_oof, gb_te
else:
    fusion, score, p_oof, p_te = "stack", s_lr, lr_oof, lr_te
th = fit_thresholds(Yi, p_oof)
pred = binarize(p_te, th)
out = SUB / f"colours5_{fusion}_{'-'.join(names)}_cv{score:.4f}_{date.today()}.csv"
write_submission(out, pred, colors)
ref = pd.read_csv(DATA / "y_random_zKiBR00.csv", index_col=0)
assert len(pd.read_csv(out, index_col=0)) == len(ref)
print("selected:", fusion, "| colours per product:", pred.sum(1).mean().round(2), "| written", out)
