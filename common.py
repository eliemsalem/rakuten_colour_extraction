"""Chargement des données, plis de validation croisée, métrique et seuils communs à tous les modèles.

Convention : chaque modèle écrit dans oof/<nom>_oof.npy (n_train x 19, probas hors-pli sur le train)
et oof/<nom>_test.npy (n_test x 19, moyenne des modèles des plis sur le test).
"""
import ast
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

HERE = Path(__file__).parent
DATA = HERE / "data"
OOF = HERE / "oof"
OOF.mkdir(exist_ok=True)
SUB = HERE / "submissions"
N_FOLDS = 5
SEED = 42

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(s, maxlen=None):
    """Retire balises et entités HTML ; tronque éventuellement (les descriptions sont parfois très longues)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))
    s = _WS.sub(" ", _TAG.sub(" ", s)).strip()
    return s[:maxlen] if maxlen else s


def load(caption_len=1000):
    """Renvoie train, test, Y (n_train x 19, 0/1) et la liste des couleurs (indice -> nom)."""
    tr = pd.read_csv(DATA / "X_train_12tkObq.csv", index_col=0)
    te = pd.read_csv(DATA / "X_test_gDTIJPh.csv", index_col=0)
    tags = pd.read_csv(DATA / "y_train_Q9n2dCu.csv", index_col=0)["color_tags"].map(ast.literal_eval)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags)
    for df in (tr, te):
        df["item_name"] = df["item_name"].map(clean_html)
        df["item_caption"] = df["item_caption"].map(lambda s: clean_html(s, caption_len))
        df["image"] = [str(DATA / "images" / f) for f in df["image_file_name"]]
    return tr, te, Y, list(mlb.classes_)


def folds(n):
    """Liste de (train_idx, val_idx), identique pour tous les modèles."""
    return list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n)))


def wf1(Y, P):
    """Métrique du challenge : F1 pondéré sur les étiquettes binarisées."""
    return f1_score(Y, P, average="weighted", zero_division=0)


def binarize(probs, thr):
    """Seuil par couleur, et au moins une couleur par produit (la plus probable)."""
    P = (probs >= thr).astype(int)
    P[np.arange(len(P)), probs.argmax(1)] = 1
    return P


def fit_thresholds(Y, probs, grid=np.arange(0.05, 0.80, 0.01)):
    """Un seuil par couleur : le F1 pondéré est la moyenne des F1 par couleur, donc on optimise chacun à part."""
    thr = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        y, p = Y[:, j], probs[:, j]
        tp = np.array([(y & (p >= t)).sum() for t in grid])
        npred = np.array([(p >= t).sum() for t in grid])
        f1 = 2 * tp / (npred + y.sum() + 1e-9)
        thr[j] = grid[f1.argmax()]
    return thr


def honest_score(Y, probs, seed=SEED):
    """Seuils réglés sur une moitié, score mesuré sur l'autre (et inversement), moyenne des deux."""
    idx = np.random.default_rng(seed).permutation(len(Y))
    a, b = idx[: len(Y) // 2], idx[len(Y) // 2:]
    s = [wf1(Y[v], binarize(probs[v], fit_thresholds(Y[t], probs[t]))) for t, v in ((a, b), (b, a))]
    return float(np.mean(s))


def save_preds(name, oof, test):
    np.save(OOF / f"{name}_oof.npy", oof.astype(np.float32))
    np.save(OOF / f"{name}_test.npy", test.astype(np.float32))


def write_submission(path, P, colors):
    """Format de 2021 : ',color_tags' avec index, une liste Python par ligne."""
    tags = [str([colors[j] for j in np.flatnonzero(row)]) for row in P]
    pd.DataFrame({"color_tags": tags}).to_csv(path)
