"""Data loading, metric, per-colour thresholds and submission format shared by the local scripts.

Out-of-fold files written by the Colab notebook: oof/oof5/<model>_f<k>.npz with
"val" (fold k rows x 19 colours) and "test" (n_test x 19 colours).
"""
import ast
import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

HERE = Path(__file__).parent
DATA = HERE / "data"
OOF = HERE / "oof"
OOF.mkdir(exist_ok=True)
SUB = HERE / "submissions"
SUB.mkdir(exist_ok=True)

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(s, maxlen=None):
    """Removes HTML tags and entities; optionally truncates (some descriptions are very long)."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))
    s = _WS.sub(" ", _TAG.sub(" ", s)).strip()
    return s[:maxlen] if maxlen else s


def load(caption_len=1000):
    """Returns train, test, Y (n_train x 19, 0/1) and the list of colours (index -> name)."""
    tr = pd.read_csv(DATA / "X_train_12tkObq.csv", index_col=0)
    te = pd.read_csv(DATA / "X_test_gDTIJPh.csv", index_col=0)
    tags = pd.read_csv(DATA / "y_train_Q9n2dCu.csv", index_col=0)["color_tags"].map(ast.literal_eval)
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(tags)
    for df in (tr, te):
        df["item_name"] = df["item_name"].map(clean_html)
        df["item_caption"] = df["item_caption"].map(lambda s: clean_html(s, caption_len))
    return tr, te, Y, list(mlb.classes_)


def wf1(Y, P):
    """Challenge metric: weighted F1 over the binarised labels."""
    return f1_score(Y, P, average="weighted", zero_division=0)


def binarize(probs, thr):
    """One threshold per colour, and at least one colour per product (the most likely one)."""
    P = (probs >= thr).astype(int)
    P[np.arange(len(P)), probs.argmax(1)] = 1
    return P


def fit_thresholds(Y, probs, grid=np.arange(0.05, 0.80, 0.01)):
    """One threshold per colour: the weighted F1 is a mean of per-colour F1 scores, so each is tuned separately."""
    thr = np.empty(Y.shape[1])
    for j in range(Y.shape[1]):
        y, p = Y[:, j], probs[:, j]
        tp = np.array([(y & (p >= t)).sum() for t in grid])
        npred = np.array([(p >= t).sum() for t in grid])
        f1 = 2 * tp / (npred + y.sum() + 1e-9)
        thr[j] = grid[f1.argmax()]
    return thr


def write_submission(path, P, colors):
    """Challenge format: ',color_tags' with the row index, one Python list per row."""
    tags = [str([colors[j] for j in np.flatnonzero(row)]) for row in P]
    pd.DataFrame({"color_tags": tags}).to_csv(path)
