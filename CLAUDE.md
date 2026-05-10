# CLAUDE.md — rules for Claude sessions on this project

Draft. Promote to `/CLAUDE.md` at the repo root once we initialise git.

## Project context

End-to-end in-silico screening pipeline (sequence/PDB + SMILES → folded structure → docked poses → ranked shortlist). Notebook-driven (must work in Colab, Windows, Mac), with reusable Python modules under `src/aidd/`. Built by lifting working code from a Leiden/CDD AI-in-Drug-Discovery 2025 course (now under `_archive/`).

The user is not a software engineer by training — favour clarity over cleverness, name things from a chemist's perspective ("ligand", "pose", "binding site"), and link to the relevant course notebook in `_archive/` when adapting code from it.

## Stack & platform rules

- **Cross-platform is non-negotiable.** Every entry-point notebook must run on Colab, Windows, and macOS without code changes. No Linux-only binaries on the hot path (the bundled `_archive/.../PLANTS` binary is excluded for this reason).
- **GPU work runs in Colab.** Folding (ColabFold/AlphaFold) and DiffDock have Colab-first notebooks; CPU paths (Vina) work everywhere as a fallback.
- **Notebooks stay thin.** User-facing notebooks orchestrate; logic lives in `src/aidd/`. Aim for <30 cells per notebook with most cells being a single function call. Long inline code in a notebook is a smell — extract.
- **Cells must be re-runnable.** Idempotent: re-running a cell with the same inputs should be a no-op (check for cached output on disk before recomputing). No hidden state across cell boundaries.

## Data conventions

- All heavy / generated data lives under `data/`. The whole `data/` tree is gitignored except `data/MANIFEST.yaml` and (optionally) small reference PDBs under `data/structures/`.
- Naming: `data/derived/<target>/<stage>/<artifact>`. Example: `data/derived/erk2_4fv7/docking/poses.sdf`, `data/derived/erk2_4fv7/ifp/training_ifp.csv.gz`.
- Don't commit `.csv.gz`, `.tgz`, `.mol2`, `.sdf`, large `.pdb`, image dumps, or `.ipynb_checkpoints/`.
- The `_archive/` folder holds the original course materials read-only. Don't edit files in `_archive/`; copy what you need into `src/` or `notebooks/`.

## Code style (Python)

- Default to writing no comments. Add one only when the *why* is non-obvious.
- Type-hint public functions in `src/aidd/`. Don't bother for one-shot notebook cells.
- Prefer `pathlib.Path` over `os.path` strings.
- Logging > `print` in `src/aidd/` modules; `print` is fine in notebook cells for user-facing output.
- Don't add backwards-compat shims, feature flags, or defensive error handling for cases that can't happen. Trust internal code; only validate at external boundaries (user-supplied SMILES, file uploads).
- Don't refactor surrounding code while fixing a bug — make the smallest change that addresses the request.

## Notebook conventions

- First cell: title + one-sentence description of what the notebook does and roughly how long it takes.
- Second cell: install/import block. Use `try: import x; except: !pip install x` so the same notebook works in Colab and in a pre-set-up local env.
- Third cell: detect the environment (`is_colab = "google.colab" in sys.modules`) and set platform-specific paths.
- Use `tqdm` for any loop over ligands/poses.
- Use `py3Dmol` (not nglview) for in-notebook 3D — works reliably in Colab and JupyterLab without extension dances. Pattern is in `_archive/Week_3_Monday_Docking_and_Scoring.ipynb`.
- Avoid `nglview` unless the user explicitly asks — the JupyterLab extension setup is painful and doesn't survive Colab.

## Dependencies & environments

- Single source of truth is `environment.yml` (conda/mamba). A `requirements-colab.txt` exists for the Colab-only path where conda isn't practical — keep the two in sync.
- Pin loosely (`rdkit>=2024.3`, not `==2024.3.5`) unless we hit a known incompatibility.
- Don't add a dependency to fix something one stdlib call away.

## Don't do these without asking

- Run docking or folding on the user's machine without confirming — these can take hours and burn battery / GPU credits. Always state expected runtime first.
- Delete or move anything in `_archive/` — that's the historical record.
- `git push`, `git rebase`, `git reset --hard`, force push, or `--no-verify`. Always confirm git operations beyond local commits.
- Install Linux-only binaries or assume `apt`. The user is on Windows; cross-platform paths only.
- Re-run an already-completed expensive stage (e.g. re-dock a full library) "to be safe". Check the cached output first; ask before overwriting.

## Preferred shape of work

- Small, reviewable steps: lift one piece of course code into `src/aidd/`, write a one-cell smoke test in a notebook, commit, move on. Don't bundle five modules into a single PR-sized change.
- When picking up unfamiliar course code, before adapting it, find the markdown cell that introduces it and read it — the chemistry context is usually there.
- When in doubt about a chemistry decision (which scoring function, which fingerprint type, which conformer count), surface the trade-off in two sentences and ask, rather than picking silently.

## Open questions tracked elsewhere

See `_planning/PROJECT_PROPOSAL.md` §6 for the list of design questions still pending user input (target choice, docker choice, folding tool, env manager, storage strategy, repo destination, archive location).
