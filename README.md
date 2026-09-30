# Two-Tower Movie Recommender

A retrieval-based recommender system built on MovieLens 100K: a two-tower
neural network trained with PyTorch, served through a FAISS similarity
index, exposed via FastAPI, and demoed with a Streamlit UI.

The headline result: under a controlled evaluation protocol, the
two-tower model more than doubles a from-scratch matrix factorization
baseline's Hit@10 (56.6% vs. 27.7%) and outperforms a pure popularity
baseline on every reported ranking metric while covering 9.5× more of
the catalog.

```
MovieLens 100K
      ↓
Temporal per-user split (fit_train / val / test)
      ↓
User & movie feature engineering
      ↓
Two-Tower model (PyTorch) — full-catalog softmax, early-stopped on val
      ↓
FAISS index (cosine similarity via inner product on L2-normalized vectors)
      ↓
FastAPI serving layer
      ↓
Streamlit demo UI
```

## Results

Every model below is evaluated on the same 1,582-movie vocabulary,
the same "seen" mask (items present in the user's development history
before test — `fit_train` + `val` combined), and the same "relevant"
definition (test rating ≥ 4, restricted to movies present in the
fit-train vocabulary) — so these numbers are directly comparable to
each other, not just impressive-looking in isolation.

| Model | Hit@10 | NDCG@10 | Recall@10 | Coverage@10 |
|---|---|---|---|---|
| Random | 8.5% | 0.010 | 0.007 | 99.7% |
| Popularity | 39.3% | 0.092 | 0.068 | 4.7% |
| Matrix Factorization (biased, from scratch) | 27.7% | 0.055 | 0.041 | 16.2% |
| **Two-Tower (this project)** | **56.6%** | **0.146** | **0.130** | **44.6%** |

Coverage@10 is the fraction of the catalog that ever appears in *any*
user's top-10 — a check against a model that just learns to recommend
whatever's popular. Two-Tower's 44.6% vs. Popularity's 4.7% indicates
that the model's recommendations are substantially less concentrated
than the popularity baseline. (The per-user sanity checks below provide
additional, qualitative evidence of taste-specific behavior.)

## The debugging story

The first working version of this model scored **2.54% Hit@10** —
*worse than random*. Getting from there to 56.6% involved finding and
fixing several distinct bugs, each worth naming because each is a
common failure mode in retrieval systems, not a one-off mistake:

1. **Missing logQ correction in the in-batch loss.** In-batch negative
   sampling draws negatives in proportion to item popularity. Without
   correcting for that, the model's optimum becomes a PMI-style score
   that actively *penalizes* popular items — the opposite of what you
   want. Switched to full-catalog softmax (cheap here, since the
   catalog is only ~1,600 items) to sidestep sampling bias entirely.
2. **Unmasked false negatives in-batch.** When the same user appeared
   multiple times in a batch with different positive movies, the
   positive movie from one row was treated as a negative for another
   row. This silently taught the model to push known positives apart.
   Fixed by masking known positive user–movie pairs within the batch.
3. **ReLU immediately before L2 normalization.** This confines every
   embedding to the positive orthant, crushing cosine similarity into
   a narrow, uninformative band. Removed the final activation.
4. **Age bucket encoded as a raw integer.** The model was treating
   "bucket 5" as five times "bucket 1." One-hot encoded instead.
5. **Vocabulary built from the wrong split.** `movie_to_idx` was
   originally built from all of `train` (fit_train + val combined),
   which quietly gave validation a more complete catalog than the
   model would ever see at real training time — making validation an
   optimistic preview relative to test. Rebuilt so the vocabulary
   comes from `fit_train` only, so val (like test) has to contend with
   items the model has never seen.
6. **Baseline comparison wasn't apples-to-apples.** The original MF
   baseline was trained and evaluated on a different vocabulary
   (1,612 movies) than the two-tower model (1,582, after the fix
   above). Retrained MF from scratch on the identical vocabulary and
   evaluation protocol before trusting any comparison between them.

The loss curve is a useful diagnostic in hindsight: the original broken
run plateaued at **~5.52**, which is close to `ln(256)`, the
cross-entropy expected from uniform guessing among 256 candidates. That
number alone was the tell that something structural, not just
under-training, was wrong.

## Architecture

**User tower:** user ID embedding (32-d) + one-hot age bucket (6) +
one-hot occupation (21) → 2-layer MLP → 32-d L2-normalized output.

**Movie tower:** movie ID embedding (32-d) + genre multi-hot (19) +
normalized release year (1) → 2-layer MLP → 32-d L2-normalized output.

Similarity is cosine (dot product of normalized vectors). Training uses
full-catalog softmax cross-entropy (positives = ratings ≥ 4), with a
masked, logQ-corrected in-batch variant also implemented for comparison
on larger catalogs where full-softmax wouldn't be tractable.

Gender and ZIP code were deliberately excluded from user features.

## Data protocol

- Per-user **temporal** split: each user's ratings sorted by timestamp;
  last 20% → test; within the remaining 80%, the last 10% →
  validation and the rest → `fit_train`.
- ID vocabulary (`user_to_idx`, `movie_to_idx`) built from `fit_train`
  **only** — validation and test can and do contain movies the model
  has never seen (30 unique in val, 83 in test), mirroring real
  deployment rather than hiding the cold-start problem.
- Early stopping uses validation NDCG@10; the test set is reserved for
  final evaluation and is not used for model selection.

## Known limitations

- **Cold start:** 83 movies in the test period never appear in
  `fit_train` and have no learned ID embedding — they're excluded from
  the retrieval candidate set entirely. A production version would need
  a content-only fallback path for new items.
- **Coverage vs. precision tradeoff unexplored:** no systematic sweep
  of temperature or embedding dimension yet — the current numbers are
  from the first configuration that passed all correctness checks, not
  a tuned optimum.
- **Popularity bias:** no explicit popularity-debiasing objective was
  implemented; coverage is reported to make recommendation
  concentration visible.

## Project structure

```
data/
  preprocess.py      # temporal split, vocab, feature engineering
models/
  two_tower.py        # Tower architecture + loss functions
  metrics.py           # shared ranking-metric harness (rank_metrics, baselines)
  train.py              # training loop, early stopping on validation
  evaluate.py            # final test evaluation + baselines
  mf_fair.py              # MF baseline, retrained on the identical vocabulary
  build_index.py            # builds the FAISS index from a trained checkpoint
  sanity_check.py             # qualitative check: titles/genres per user
app/
  schemas.py           # request/response models
  engine.py              # caches FAISS index, delegates to build_index.recommend()
  main.py                  # FastAPI routes
ui/
  streamlit_app.py         # demo UI
artifacts/
  two_tower.pt              # trained model checkpoint
  movie_index.faiss           # FAISS index
  user_vectors.npy             # cached user embeddings
```
## Dataset

This project uses the MovieLens 100K dataset.

1. Download **MovieLens 100K** from the official GroupLens dataset page:
   https://grouplens.org/datasets/movielens/100k/

2. Extract the dataset.

3. Place the extracted `ml-100k` folder here:

```text
data/raw/ml-100k/
```
## Running it

```bash
pip install -r requirements.txt

# 1. Build the data pipeline (temporal split, vocab, features)
python -m data.preprocess

# 2. Train the two-tower model (early-stops on validation)
python -m models.train

# 3. Train the fair MF baseline on the same fit-train vocabulary
python -m models.mf_fair

# 4. Evaluate the two-tower model and random/popularity baselines
python -m models.evaluate

# 5. Build the FAISS index from the trained checkpoint
python -m models.build_index

# 6. (optional) qualitative sanity check across a few users
python -m models.sanity_check

# 7. Serve the model
uvicorn app.main:app --reload

# 8. In a second terminal, launch the demo UI
streamlit run ui/streamlit_app.py
```

## Tech stack

PyTorch · FAISS · FastAPI · Streamlit · pandas/NumPy · MovieLens 100K
