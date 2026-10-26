# Multi-label colour extraction from Japanese e-commerce listings (Rakuten × ENS Challenge Data #59)

Rakuten Institute of Technology asks for the **colours of a product** in the Rakuten Ichiba (Japan) catalogue,
predicted from its image, its title and its description, both in Japanese. A product can have several of the
19 colours, and the metric is the weighted F1 score over the binarised labels.

**Result: public score 0.7929**, up from 0.7002 for my text-only submission of 2021 (the organisers' image-only
benchmark scores 0.5041). The best public score on the leaderboard on 2026-10-23 was 0.7756.

## Data

The challenge data is **not in this repository**: it is confidential and may only be used for the challenge.
It can be downloaded, after creating a free account, from the challenge page:
**[ENS Challenge Data #59, Rakuten Multi-modal Colour Extraction](https://challengedata.ens.fr/participants/challenges/59/)**.

| File (name as downloaded) | Put it in | Content |
|---|---|---|
| `X_train_12tkObq.csv` | Google Drive `rakuten_couleurs/` and `data/` | 212,120 products: image file name, title (`item_name`), description (`item_caption`) |
| `y_train_Q9n2dCu.csv` | Google Drive `rakuten_couleurs/` and `data/` | colour lists, e.g. `['White', 'Green', 'Black']` |
| `X_test_gDTIJPh.csv` | Google Drive `rakuten_couleurs/` and `data/` | 37,347 test products |
| `images_NFkV58V.tar` | Google Drive `rakuten_couleurs/` | one image per product |
| `y_random_zKiBR00.csv` | `data/` | sample submission, used to check the row order |

## Approach

1. **A validation I can trust before anything else.** Every model is trained in 5 folds, so each training product
   gets an out-of-fold prediction. Scores are measured on these 212,120 predictions with the per-colour thresholds
   tuned on one half and the F1 measured on the other half. This estimate has stayed 0.5 to 0.8 points below the
   public score at every submission.
2. **Three base models with different views of the product:**
   - image: frozen SigLIP so400m embeddings plus 19 zero-shot colour scores ("a photo of a red product"),
     then a multi-label MLP head;
   - text: character n-gram TF-IDF (Japanese has no spaces) with one logistic regression per colour;
   - text: `tohoku-nlp/bert-base-japanese-v3` fine-tuned with a binary cross-entropy loss, 3 epochs, 256 tokens.
3. **Stacking** on the out-of-fold logits: one meta-model per colour.
4. **Catalogue neighbours.** Image file names look like `<shop>_<item>_1.jpg`. Sorted by shop then item number,
   a product has exactly the same colours as the next product of the same shop in 16.6% of cases (3.9% for two
   random products). The meta-model gets the mean colours of the 5 previous and 5 next training products of the
   same shop, the product itself excluded, and the shop's usual colours computed out-of-fold.
5. **Decision:** one threshold per colour, since the weighted F1 is a frequency-weighted mean of per-colour F1
   scores, and at least one colour per product.

## Results

| Step | Validation | Public score |
|---|---|---|
| 2021: Japanese BERT on text only, one threshold (0.28) for all colours | — | 0.7002 |
| v1: SigLIP head + BERT (2 epochs, 128 tokens) + TF-IDF, weighted average, per-colour thresholds | 0.7520 (10% holdout) | 0.7576 |
| v2: BERT 3 epochs / 256 tokens, logistic-regression stacker trained on the holdout | 0.7613 (holdout) | 0.7687 |
| v3: all base models in 5 folds, stacker trained on the 212,120 out-of-fold rows | 0.7653 (5-fold) | 0.7732 |
| **v4: + catalogue neighbours and shop colour prior, LightGBM stacker** | **0.7850** (5-fold) | **0.7929** |

Single models, 5-fold: image 0.6815, TF-IDF 0.6828, BERT 0.6977. Each model alone is around 0.68 to 0.70;
together they reach 0.7653, so the three views are complementary.

What did not help:
- **Qwen2.5-7B fine-tuned with QLoRA** (title only, 64 tokens, 1 epoch): 0.6113 on fold 0, against 0.6993 for
  BERT on the same fold. The run was stopped after this fold.
- **A stacker trained on a 10% holdout** (v2) instead of the full out-of-fold predictions (v3): v3 gained
  0.45 points on the public score with the same base models apart from the 5-fold training.

## Repository

| File | Role |
|---|---|
| `colab/rakuten_couleurs_5plis.ipynb` | Colab notebook: SigLIP embeddings, the three base models in 5 folds (saved fold by fold to Google Drive, resumable), stacking with and without neighbours (XGBoost), submission |
| `stack_voisins.py` | Local step 4 with LightGBM: reads the out-of-fold files, compares the stacker with and without neighbours, writes the submission (this produced the 0.7929 file) |
| `common.py` | Data loading, label binarisation, metric, per-colour thresholds, submission format |
| `requirements.txt` | Packages for the local step (Colab installs its own) |

Code comments and notebook text are in French.

## Reproduce

1. Put the challenge files in a Google Drive folder `rakuten_couleurs/` (see [Data](#data)).
2. Open `colab/rakuten_couleurs_5plis.ipynb` in Colab with an A100 GPU and run all cells. Measured durations on
   A100: about 23 min per BERT fold, about 10 min per TF-IDF fold; the SigLIP embeddings took 34 min on an L4.
   The out-of-fold files go to `Drive/rakuten_couleurs/out/oof5/`.
3. Locally, copy that folder to `oof/oof5/` and the CSV files to `data/`, then:
   ```
   pip install -r requirements.txt
   python stack_voisins.py
   ```
   The submission is written to `submissions/`.

## Limitations

- **The neighbour features describe how the catalogue was assembled, not the product.** Shops list products in
  batches with close item numbers, and the test products are interleaved with the training ones: 98% of test
  products have a training neighbour from the same shop. On their own, the neighbours reach only 0.43 F1
  (holdout), but they add 2 points to the stacker. They would not help for a product from an unseen shop.
- Scores are public-leaderboard scores. The private leaderboard is only updated twice a year.
- The Colab notebook's neighbour step (XGBoost) was tested end to end on a 3,000-row sample only. The submitted
  file came from `stack_voisins.py` (LightGBM) run on the full out-of-fold files.
- My files have 212,120 training and 37,347 test products, while the challenge page announces 212,659 and 37,528.
  I have not checked whether the data was updated after 2021.
