# CLAUDE.md — rules for Claude sessions on this project

## Project context

End-to-end in-silico screening pipeline (sequence/PDB + SMILES → folded structure → docked poses → ranked shortlist). Notebook-driven (must work in Colab, Windows, Mac), with reusable Python modules under `src/aidd/`. Built by lifting working code from a Leiden/CDD AI-in-Drug-Discovery 2025 course (now under `_archive/`).

The user is Natallia, a medical doctor in oncology / cancer genetics; not a software engineer or ML researcher. Favour clarity over cleverness, name things from a chemist's perspective ("ligand", "pose", "binding site"), and link to the relevant course notebook in `_archive/` when adapting code from it.

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

## Notebook workflow (mandatory)

**The source of truth is a Python builder script, not the .ipynb.** This is non-negotiable.

| File | Role |
|---|---|
| `notebooks/_build_<name>.py` | **Source.** All notebook content lives here as Python literals (markdown strings, code-cell strings, helper functions, the call to `nbformat.write`). Always tracked in git. |
| `notebooks/<name>.ipynb` | **Generated artefact.** Tracked in git with cell outputs *after the user runs it*, so reviewers see results without having to execute anything. Never authored directly. |

Workflow:

1. Edit `_build_<name>.py`.
2. Run `python notebooks/_build_<name>.py` to regenerate the `.ipynb`.
3. The user opens `<name>.ipynb` in VS Code / Jupyter / Colab and runs it.

Rules:

- **Never edit `<name>.ipynb` directly.** Edits there get overwritten on the next regen. Always change the builder.
- **Regenerate before committing** whenever the builder changes, so the committed `.py` and `.ipynb` are in sync. The user should not be the one running the regenerator.
- **Do not strip outputs** with `nbstripout` or any pre-commit hook. Outputs are intentionally preserved in git.
- **If the `.ipynb` has diverged from what the builder would produce** (because the user experimented in Jupyter), do not silently overwrite. Surface the diff and ask whether to (a) regenerate and lose the experiments, or (b) port them back into the builder first.
- Use `nbformat` (already installed via Jupyter) for the builder. No Jupytext / MyST / other alternatives.

Cells are emitted via shared helpers in `notebooks/_nb_helpers.py`:

- `markdown("""…""")` for markdown cells
- `code("""…""")` for code cells
- `notebook(*cells)` for assembly with our kernel metadata
- `save(nb, path)` for writing

**Cells appear in the builder in narrative order** — the builder reads top-down as the rendered notebook does, so there is no separate "content constants → assembly list" indirection.

## Notebook conventions (structural)

- First cell: title + one-paragraph plain-language description, learning objectives, audience, prerequisites (see *Notebook pedagogy* below).
- Setup cells: install/import block (`try: import x; except: !pip install x`) followed by env detection (`is_colab = "google.colab" in sys.modules`) and path setup. Include `%load_ext autoreload; %autoreload 2` so iterative edits to `src/aidd/` don't require kernel restarts.
- Use `tqdm` for any loop over ligands/poses.
- Use `py3Dmol` (not nglview) for in-notebook 3D — works reliably in Colab, VS Code, and JupyterLab without extension dances. Pattern is in `_archive/Week_3_Monday_Docking_and_Scoring.ipynb`.
- Avoid `nglview` unless the user explicitly asks — the JupyterLab extension setup is painful and doesn't survive Colab.
- **Colab artefacts must be turn-key, not sketches.** Any cell sequence, probe, verification step, or notebook designed for Natallia to run on Colab must be a complete click-Connect-Run-all artefact: include install + repo clone + Drive mount + all path/variable definitions + the actual work + the print statements that emit results. No assumed prior state ("run nb 07's setup first then paste these cells" is the anti-pattern). Treat every Colab artefact as if a student is running it cold. Full rationale + anti-pattern example in `feedback_turnkey_colab_outputs.md` memory file.

## Notebook pedagogy (mandatory)

**The notebooks teach as much as they compute.** They are read by:
- Colleagues running the pipeline in parallel — mixed biomed / data / ML backgrounds.
- Non-native English speakers — language must be simple.
- Bachelor / Master students writing their thesis — entry-level accessible.
- Department heads, senior researchers, funding-agency auditors — must look scientifically sound and reflect state-of-the-art methods.

The Leiden/ULLA course archived in `_archive/` is the style reference: pedagogical AND state-of-the-art at the same time.

**Every notebook follows this shape:**

1. **Title cell.** Project line + one-paragraph plain-language description of what the notebook does. **Learning objectives** (4–6 bullets). **Audience** ("you'll get value if you are…"). **Prerequisites** (env / data / prior notebooks).
2. **"Key terms" block** near the top (or inline definitions later). Define every jargon term and abbreviation on first use anywhere in the notebook.
3. **For each working section**, the shape is:
   - **Markdown — Background.** What this concept is (biomedically AND technically), why it matters clinically/scientifically, how it fits the pipeline. 2–4 short paragraphs max.
   - **Markdown — What this cell does.** One short sentence orienting before the compute. Vary the lead-in; don't write "In this cell we will…" every time.
   - **Code cell.**
   - **Markdown — Interpreting the output.** What to look at, what the numbers mean, what's "good" vs "bad", state-of-the-art thresholds where relevant, common pitfalls.
4. **Recap cell** at the bottom: biomedical takeaway (one short paragraph), technical takeaway (one short paragraph), pointer to the next notebook in the pipeline, 2–3 further-reading references (paper DOIs or canonical docs, not random blog posts).

**Writing rules:**

- **Plain language. No jargon without definition.** "Pi-stacking" gets a one-line explanation the first time it appears.
- Short sentences. Active voice. Avoid idioms and figures of speech (the audience includes ESL readers).
- No filler ("Note that…", "It is important to mention that…", "Basically…"). Just say it.
- Expand every abbreviation on first use: "AlphaFold (AF)", "interaction fingerprint (IFP)".
- Where possible, give a clinical or wet-lab analogue ("docking is the in-silico version of a binding assay").
- Acknowledge the state of the art **and** what we're using **and** why. Don't oversell our choices; mention limits honestly.
- Don't talk down. Assume a smart non-expert — give them the bridge, not a lecture.
- **Teaching goes in markdown cells, not code comments.** The `Code style (Python)` rules above still apply to the cells themselves: minimal inline comments, clean code.

### Notebook 99 (production runner) is *also* a teaching notebook

`99_screen_library.ipynb` is what reviewers, professors, and grant funders will read. It still teaches, just tighter than 00–06:

- Background blocks: 1–2 short paragraphs per stage (vs 3–4 in teaching notebooks).
- No deep glossary table — expand abbreviations on first use inline.
- Every method gets a one-line description **with a citation (paper DOI)** so reviewers can verify the methodological pedigree.
- A "Methods summary" cell near the top a funder can read in 60 seconds.
- Recap is the executive summary: result + confidence + caveats + provenance links back to notebooks 00–06.

What 99 does *not* do: redefine concepts that the teaching notebooks already cover. Reviewers drill into 00–06 for detail.

## Dependencies & environments

- Single source of truth is `environment.yml` (conda/mamba). A `requirements-colab.txt` exists for the Colab-only path where conda isn't practical — keep the two in sync.
- Pin loosely (`rdkit>=2024.3`, not `==2024.3.5`) unless we hit a known incompatibility.
- Don't add a dependency to fix something one stdlib call away.
- **On Natallia's Windows machine, never call `python` / `py` / `python3` directly from tool-call shells** — they're not on PATH because tool calls don't inherit `conda activate`. Use the full interpreter path `C:\Users\MarcoHernandez\.conda\envs\aidd\python.exe` or `C:\ProgramData\miniconda3\Scripts\conda.exe run -n aidd python ...`. Project env is `aidd`. Full rationale + verification snippets in the `feedback_python_invocation_on_windows.md` memory file.

## Don't do these without asking

- Run docking or folding on the user's machine without confirming — these can take hours and burn battery / GPU credits. Always state expected runtime first.
- Delete or move anything in `_archive/` — that's the historical record.
- `git push`, `git rebase`, `git reset --hard`, force push, or `--no-verify`. Always confirm git operations beyond local commits.
- Install Linux-only binaries or assume `apt`. The user is on Windows; cross-platform paths only.
- Re-run an already-completed expensive stage (e.g. re-dock a full library) "to be safe". Check the cached output first; ask before overwriting.

## Step-closing approval gates (mandatory)

For any commit/push that **closes a numbered step** in `_planning/PROJECT_PROPOSAL.md` § *First-week deliverables* — i.e. one that:

- Updates a `notebooks/README.md` status row from anything to `✅ done`, OR
- Lands the empirical numbers that satisfy a step's done-signal (ROC-AUC, redock RMSD, PoseBusters pass rate, smoke-test outputs, etc.), OR
- Declares any phase of the 13-step plan "verified",

the lead implementation agent **must stop before `git push`** and post a status update containing:

1. **The empirical findings** (the numbers that close the step, as a short table or bullet list).
2. **`git diff --stat`** for everything that's about to be pushed.
3. **The proposed commit message(s)**, verbatim — title plus body.
4. **A one-line ask**: *"approved to push?"*

Then **wait for an explicit `yes` / `approved` / `push it`** before running `git push`. If the reviewer asks for changes (different message, split commits, hold a README update), apply them and re-post for approval.

Routine intermediate commits during a step build (writing a cell, fixing a typo, tailing a log, debugging an install) **do not** need this gate. They commit freely; the push waits for the end-of-round approval. Planning-doc edits in `_planning/*.md` made by the reviewer follow a separate `--diff` review flow and don't go through this gate either.

**Why this rule exists** (and why it's stricter than the general commit-hygiene rule): step-closing pushes land verified status on the visible branch, embed empirical numbers in commit messages that future reviewers and grant materials will reference, and tell downstream agents what's done. A bad step-closing push (a "verified" row that isn't, wrong numbers in a commit, an architectural concession buried in a feature commit) is expensive to correct after the fact and pollutes the git log. One short pre-push approval turn is cheaper than the alternative.

## Preferred shape of work

- Small, reviewable steps: lift one piece of course code into `src/aidd/`, write a one-cell smoke test in a notebook, commit, move on. Don't bundle five modules into a single PR-sized change.
- When picking up unfamiliar course code, before adapting it, find the markdown cell that introduces it and read it — the chemistry context is usually there.
- When in doubt about a chemistry decision (which scoring function, which fingerprint type, which conformer count), surface the trade-off in two sentences and ask, rather than picking silently.

## Open questions tracked elsewhere

See `_planning/PROJECT_PROPOSAL.md` §6 for the list of design questions still pending user input (target choice, docker choice, folding tool, env manager, storage strategy, repo destination, archive location).
