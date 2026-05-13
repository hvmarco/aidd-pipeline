# Known issues — post-step-N follow-ups

Issues caught while closing a step that don't block the step itself but require a fix before some downstream consumer can rely on the data. Each entry: **bug**, **symptom**, **honest number** (if applicable), **fix scope**, **priority** (blocking vs. background).

## Open

### Rescorer `rescorer_rf_proba` in `scored_poses.parquet` is data-leaked

**Surfaced:** 2026-05-13, during step 11's evaluation cell (notebook 05).

**Bug.** Notebook 04 currently does the following at step 10's closing:

```python
final_rf = make_rescorer("rf", smote=True, seed=42, n_estimators=300)
final_rf.fit(X, y)                                # trained on ALL labelled compounds
all_proba_rf  = final_rf.predict_proba(X)[:, 1]   # predicts on ALL labelled compounds
scored["rescorer_rf_proba"] = all_proba_rf
scored["in_random_test_fold"]   = ...             # fold membership flags (correct)
scored["in_scaffold_test_fold"] = ...
```

`rescorer_rf_proba` is therefore the prediction of a model that has seen its own test-fold rows during training. The fold-membership columns are correct, but the *probability* column is not split-aware.

**Symptom.** Any downstream notebook that filters `scored_poses.parquet` to a test fold and computes a held-out metric on `rescorer_rf_proba` will see **AUC = 1.000** — the model perfectly classifies the data it was trained on. Notebook 05's Section 8 evaluation cell hit this on its first run; we caught it because the result was implausibly perfect alongside a Boltz-2 AUC of 0.65.

**Honest number.** Notebook 04's own held-out evaluation (computed correctly via `_train_and_eval`, where the model is trained only on `train_idx` and predicted on `test_idx`) reports:

- Random-split RF AUC: **0.638** (per step-10 README status row)
- Scaffold-split RF AUC: **0.663** (the headline number from step-10's closing)

These are the real performance numbers. Anything close to 1.000 in a downstream notebook is the leak.

**Fix scope.** Update `notebooks/_build_04_score_classical.py` such that `scored_poses.parquet` carries **out-of-fold (OOF) predictions** in a separate column — i.e., for each compound, store the probability assigned by a model that did NOT see that compound during training. Two implementation options:

- (a) **Cross-validated OOF.** Run an N-fold CV (e.g., the same scaffold splitter, but N=5 splits), and for each fold's test set, record the predict_proba from the model trained on the other N−1 folds. Saves as `rescorer_rf_proba_oof`. Standard QSAR practice; gives one OOF prediction per compound where every prediction is honest.
- (b) **Simpler: save train/test predictions separately.** Keep the existing single-split predictions but flag the column explicitly: `rescorer_rf_proba_train_on_all` (the current leaked column, useful only for chemistry inspection) plus `rescorer_rf_proba_test_only` (the held-out predictions from `_train_and_eval`). Downstream notebooks use the latter.

Option (a) is cleaner and aligns with how the consensus shortlist (notebook 06) will need to evaluate the rescorer column across the *full* 413-compound cohort. Option (b) is faster to implement but only useful for the existing 80/20 split.

**Priority.** **Not blocking step 11 close** (Boltz-2's done signal is on Boltz-2's columns, which are honest). **Blocking** any honest head-to-head comparison plot between rescorer and Boltz-2 in notebook 06 (consensus) or notebook 99 (production runner). Fix before notebook 06 closes.

**Workaround until fixed.** Notebook 05's Section 8 metrics table labels the rescorer row "LEAKED" and includes a markdown note pointing here. Any reporting / talk / paper material should quote the honest 0.66 scaffold AUC from step 10, never the 1.000 from step 11.

---

## Closed

*(none yet)*
