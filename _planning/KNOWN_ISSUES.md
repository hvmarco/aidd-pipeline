# Known issues — post-step-N follow-ups

Issues caught while closing a step that don't block the step itself but require a fix before some downstream consumer can rely on the data. Each entry: **bug**, **symptom**, **honest number** (if applicable), **fix scope**, **priority** (blocking vs. background).

## Open

*(none)*

---

## Closed

### Rescorer `rescorer_rf_proba` in `scored_poses.parquet` is data-leaked

**Surfaced:** 2026-05-13, during step 11's evaluation cell (notebook 05).
**Resolved:** Fixed in this commit. 5-fold scaffold OOF aggregate over all 413 labelled compounds: **RF AUC = 0.584, XGB AUC = 0.581**, with high per-fold variance (per-fold RF AUCs: 0.679, 0.404, 0.533, 0.693, 0.653 — Fold 2 is below random; Fold 4 is the highest). The OOF aggregate **supersedes step 10's single 80/20 scaffold split number (0.663)** as the rescorer's scaffold-generalisation estimate; both were honest realisations, but the 5-fold aggregate covers every compound and every scaffold, whereas the single split landed at a high realisation in this wide distribution. Done-signal still passes — margin over baseline 0.478 is **+0.106** for RF, **+0.103** for XGB. The per-fold spread is the scientifically informative second number: the rescorer generalises *unevenly* across chemotypes (a wet-lab-relevant honesty signal, exactly what scaffold splits exist to expose).

**Bug.** Notebook 04 did the following at step 10's closing:

```python
final_rf = make_rescorer("rf", smote=True, seed=42, n_estimators=300)
final_rf.fit(X, y)                                # trained on ALL labelled compounds
all_proba_rf  = final_rf.predict_proba(X)[:, 1]   # predicts on ALL labelled compounds
scored["rescorer_rf_proba"] = all_proba_rf
scored["in_random_test_fold"]   = ...             # fold membership flags (correct)
scored["in_scaffold_test_fold"] = ...
```

`rescorer_rf_proba` was therefore the prediction of a model that had seen its own test-fold rows during training. The fold-membership columns were correct, but the *probability* column was not split-aware.

**Symptom.** Any downstream notebook that filtered `scored_poses.parquet` to a test fold and computed a held-out metric on `rescorer_rf_proba` saw **AUC = 1.000** — the model perfectly classifies the data it was trained on. Notebook 05's Section 8 evaluation cell hit this on its first run; we caught it because the result was implausibly perfect alongside a Boltz-2 AUC of 0.65.

**Honest number.** Notebook 04's own held-out evaluation (computed correctly via `_train_and_eval`, where the model is trained only on `train_idx` and predicted on `test_idx`) reported:

- Random-split RF AUC: **0.638** (per step-10 README status row)
- Scaffold-split RF AUC: **0.663** (the headline number from step-10's closing)

These are the real performance numbers. Anything close to 1.000 in a downstream notebook was the leak.

**Resolution.** `_build_04_score_classical.py` now adds a new Stage F: a 5-fold scaffold-grouped CV that produces out-of-fold (OOF) predictions for every labelled compound. `scored_poses.parquet` gains two honest columns (`rescorer_rf_proba_oof`, `rescorer_xgb_proba_oof`) consumed by `aidd.consensus.compute_consensus` (which defaults `rescorer_col="rescorer_rf_proba_oof"` and raises `KeyError` if the column is missing, preventing accidental runs against the leaked alternative). The legacy `rescorer_rf_proba` / `rescorer_xgb_proba` columns are retained explicitly for chemistry-inspection use (scoring brand-new compounds at deployment time), with a table in the save section documenting which column to use for which purpose. The saved `rescorer.pkl` is still the model trained on all data — the OOF AUC is its performance estimate, not the model itself.
