# Known issues — post-step-N follow-ups

Issues caught while closing a step that don't block the step itself but require a fix before some downstream consumer can rely on the data. Each entry: **bug**, **symptom**, **honest number** (if applicable), **fix scope**, **priority** (blocking vs. background).

## Open

### nb 09 stub IFP uses relaxed RDKit parse -- strict fix decision deferred to step 17 nb 99 design

**Surfaced:** 2026-05-17, during nb 09 Colab T4 run for step 16 closure.
**Status:** Open. Mitigation shipped (this turn); permanent fix is a step-17 Q-decision.

**Bug.** PDBFixer-produced PDBs (from `aidd.inputs.prep_receptor` + `PDBFixer.applyMutations`) occasionally have atoms RDKit's strict valence sanitization rejects -- typically an over-valent carbon where RDKit's proximity-bonding heuristic infers extra bonds between residues PDBFixer reconstructed. `Chem.MolFromPDBFile` returns `None`; `aidd.ifp.load_plf_molecule` raises. Hit on the very first Colab run of nb 09's §5 NAT2 walkthrough (NAT2 WT PDB 2PFR, post-`prep_receptor` cleanup; specifically `"Explicit valence for atom # 65 C, 5, is greater than permitted"`).

nb 08's stub PDB came from a BioPython-only sidechain surgery on an AlphaFold model -- different PDB pedigree, different bond-perception outcome, no failure. nb 09's stub uses the production `prep_receptor + PDBFixer.applyMutations` path (the same one nb 99 will use in step 17), which is the path that surfaces this.

**Mitigation (this turn).** `src/aidd/ifp.py:load_plf_molecule` now has a defensive two-step: strict parse first (unchanged behavior; nb 08 still takes the fast path); on `None`, retry with `sanitize=False` and re-sanitize without `Chem.SanitizeFlags.SANITIZE_PROPERTIES` (the valence-check flag). ProLIF receives a geometrically-correct mol with relaxed bond perception. Logs a warning when the fallback fires. Backward compatible.

**Honest trade-off.** The relaxed parse preserves geometric IFP categories (residue distance shells, H-bond geometry, hydrophobic contacts -- all rely on Cα / heavy-atom coordinates, which are unchanged) but bond-order-dependent categories may misclassify:

- Aromatic stacking detection depends on aromaticity perception, which depends on bond orders.
- Formal-charge-dependent HBDonor / HBAcceptor classification depends on valences.

For step-16 stub fixtures this is acceptable -- methodology verification, not real binding fidelity. For step-17 nb 99 production runs the strict fix becomes a methodological question.

**Strict-fix candidates for step-17 nb 99 Q-decision.** Three paths:

- **(A) Filter offending atoms post-PDBFixer.** Detect over-valent atoms in the prepped PDB and drop them before RDKit parse. Preserves most of PDBFixer's geometric output but drops information; aromatic-stacking IFP categories may still misclassify on remaining atoms whose perceived bond orders depend on the dropped atoms' neighbours.
- **(B) OpenBabel pre-sanitize.** Pipe PDBFixer output through `obabel -ipdb -opdb` (or `pybel.readfile`); OpenBabel re-perceives bonds via a different heuristic before RDKit parse. Replaces one heuristic with another -- same class of problem from a different angle. Cheap to try, modest dependency add.
- **(C) `OpenMM Modeller.addHydrogens` after PDBFixer + re-parse.** Heaviest but most methodologically correct: explicit hydrogens force consistent valence per the OpenMM force-field model. Most likely to produce a strict-parseable PDB, at the cost of an extra OpenMM step in the call chain and slightly larger PDBs.

**Fix scope.** Pick one of (A) / (B) / (C) **deferred to step 18** once nb 99's first production runs reveal whether the aromatic-stacking miscount is empirically material on real PDBs (rather than synthetic stubs). Step-17 ships nb 99 with the existing relaxed-parse fallback; the executive-summary recap documents the IFP-fidelity caveat. Measurement criterion for step 18: how often does the strict-parse fallback fire across the headline-demo target set's first production-mode runs, and how many IFP entries differ between the relaxed-parse output and a candidate strict-fix output on the same input. Update `load_plf_molecule` if the strict fix lands at the IFP-reader layer, or `prep_receptor` if it lands at the PDB-prep layer.

**Priority.** Background. Does not block step-16 closure; nb 99 ships at step 17 with the relaxed-parse fallback + an explicit caveat in the executive recap. Becomes load-bearing in step 18 once production-run IFP data lets us measure the actual miscount frequency.

---

### nb 99 `_run_target_moa` failure-path is architecturally shipped but never exercised on real data

**Surfaced:** 2026-05-18, during step-17 closure run on Colab Pro+ A100.
**Status:** Open. Step-18 follow-up.

**Bug.** The graceful-failure path that spans `aidd.co_folding.predict_complex` (raises `BoltzPredictionError` with typed `error_class`), `aidd.mutation.render_moa_html` (`is_boltz_failed` branch → partial HTML with amber `.boltz-fail` banner + collapsible log-tail `<details>` block), and nb 99's `_run_target_moa` (per-(compound, variant) `try`/`except BoltzPredictionError` → priors-only HTML + summary-CSV `error` / `error_detail` columns) was shipped at commit `926368c` (Piece 4.5b of step 17). Architecturally complete and unit-verified at module-load level (`BoltzPredictionError.__mro__` check; builder regen at expected cell count). But every (compound, variant) pair in the A100 closure run on 2026-05-18 succeeded, so the failure-path code has never executed against a real Boltz failure end-to-end. The visual rendering — amber-banner partial HTML, log-tail truncation in `<details>`, summary-CSV row with populated `error` column + 300-char-clipped `error_detail` — is therefore unvalidated in a real browser / on a real CSV consumer.

**Honest trade-off.** Code review confirms the failure-path logic is structurally correct, but "structurally correct" and "renders cleanly in a real browser" can diverge — HTML / CSS rendering bugs, character-escape edge cases in the log-tail (the boltz.log tail can contain `<`, `>`, `&`, raw newlines), and CSV-quoting issues when `error_detail` contains commas or quotes all surface only on first contact with real data. Until exercised, treat the failure-path as architecturally-shipped-but-unverified.

**Strict-fix candidates for step 18.**

- **(A) Force a synthetic Boltz failure for visual verification.** A small scratch script that calls `_run_target_moa` against a deliberately-too-large target on T4 (DPYD on T4 will OOM reliably per the 2026-05-18 closure-run-attempt that triggered this commit's `BoltzPredictionError` plumbing in the first place). Inspect the rendered HTML in a browser + the summary CSV in pandas. One Colab iteration cycle on T4 (cheap; T4 burn rate ≈ 1/3 of A100).
- **(B) Wait for the natural first occurrence.** First time an operator runs nb 99 on a smaller GPU against a large target (T4 against DPYD or UGT1A1), the failure path fires. Operator pastes back the rendered HTML; verify or patch as needed. Free, but blocks first-occurrence diagnosis until someone hits it organically.

**Fix scope.** Visual inspection of one real-data failure rendering; patch CSS / HTML-escape / CSV-escape if any rendering oddity surfaces. Likely no patches needed; the surface area is small.

**Priority.** Background. Does not block step-17 closure; the architectural surface (try/except scope, `BoltzPredictionError` MRO, summary-CSV schema bump) is correct by construction.

---

### nb 99 per-(compound, variant) HTML reports inherit Boltz-2 cofactor asymmetry

**Surfaced:** 2026-05-18, during step-17 nb 99 build (documented in nb 99 cell 27's Caveats markdown as a methods-paper-grade caveat; broken out here as an explicit known-issues entry).
**Status:** Open. Step-18 follow-up if cofactor-dependent demos enter publication-grade interpretation.

**Bug.** Boltz-2 co-folds **protein + ligand only** — cofactors are not part of the Boltz YAML schema. For cofactor-dependent demos (CYP2D6 with HEM iron; DPYD with FAD / NAD(P)+ / FMN / [4Fe-4S]; UGT1A1 with UDP-glucuronic acid in any future model), the **gnina-CNN lane** in nb 99's library mode sees the *holo*-with-cofactor receptor that `aidd.inputs.prep_receptor` produces (cofactors retained per the per-target `RECEPTOR_PREPS` preset), while the **Boltz-2 lane** sees the *apo* pocket. The two lanes therefore see different receptor states on cofactor-dependent targets.

The per-(compound, variant) HTML reports rendered by `aidd.mutation.render_moa_html` inherit this asymmetry: the Boltz-2 affinity row shows the apo-pocket prediction; the gnina-CNN row (currently `None` in nb 99 MoA mode per the `_run_target_moa` helper; would be populated by a future per-compound gnina extension) would show the holo-with-cofactor prediction. The methods-paper caveat at nb 99 cell 27 quotes this directly: *"Agreement between the lanes is stronger evidence than either alone; divergence between the lanes specifically on cofactor-dependent targets warrants careful chemistry-side review (one lane is seeing an active-site environment the other is missing)."*

**Honest trade-off.** The asymmetry is a methodological constraint of Boltz-2's input schema, **not a pipeline bug**. For non-cofactor targets (ERK2 ATP-binding-site kinases; KRAS small-GTPases without bound cofactors at the inhibitor site; NAT2 with the acetyl-CoA cofactor not required for substrate-binding-fit modelling) the asymmetry does not apply. For the cofactor-dependent demos in the headline set (CYP2D6, DPYD, UGT1A1), Boltz's affinity numbers are sub-optimally informed by the absence of the bound cofactor.

**Strict-fix candidates for step 18.**

- **(A) Wait for Boltz-2 to add cofactor schema support.** Upstream feature request; not in the project's control. Boltz-3 or a later release may close the gap.
- **(B) Add a per-compound gnina lane to nb 99's MoA mode.** Per-(compound, variant) gnina docking gives the holo-with-cofactor affinity number explicitly; `render_moa_html` then renders both lanes side-by-side in the scores table (already supports `gnina_cnn_affinity` row; currently `None`); divergence between the lanes becomes the explicit read-out. ~80 lines added to `_run_target_moa` (single-compound `dock_library` invocation + parsed score extracted into the `scores.gnina_cnn_affinity` field of the record dict). The `posebusters_pass` row in the HTML scores table is similarly currently `None` and would be populated in the same step.
- **(C) For high-divergence cofactor-dependent targets, run a covalent / MD validation outside the pipeline.** Out of scope for the in-silico screening pipeline; same scope-boundary as the catalysis-rate prediction caveat in nb 99's "What this notebook does NOT predict" block.

**Fix scope.** (B) is the natural step-18 follow-up if cofactor-dependent demos enter publication-grade interpretation. The asymmetry is acceptable for the in-silico hypothesis-generation use case the pipeline is designed for (per `_planning/MECHANISM_OF_ACTION_SCOPE.md`'s "Honesty: what the pipeline does and does NOT predict" section).

**Priority.** Background. Does not block step-17 closure; nb 99 cell 27 documents the asymmetry as a methods-paper caveat reviewers will see.

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
