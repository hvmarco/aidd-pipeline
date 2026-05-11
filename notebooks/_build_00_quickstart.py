"""Builder for 00_quickstart.ipynb.

Source of truth for the quickstart notebook. Cells appear below in the order
they will render. Never edit the .ipynb directly — see ``CLAUDE.md`` §
*Notebook workflow*.

Regenerate:
    python notebooks/_build_00_quickstart.py
"""

from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _nb_helpers import code, markdown, notebook, save  # noqa: E402

NOTEBOOK_PATH = HERE / "00_quickstart.ipynb"


def build() -> None:
    nb = notebook(
        markdown("""
# 00 — Quickstart: protein–ligand interaction fingerprints on ERK2

**aidd-pipeline · Notebook 0 of the screening workflow**

This is the simplest end-to-end demonstration in the pipeline. We take a known protein–ligand complex from the Protein Data Bank — the kinase **ERK2** bound to its co-crystallised small-molecule inhibitor — and we ask two questions:

1. *Where on the protein does the ligand sit, and which residues does it touch?*
2. *What kinds of chemical interactions hold the ligand in place?*

The answers come out as a small data table called an **interaction fingerprint (IFP)** plus three figures (a 3-D viewer, a 2-D diagram, and a 3-D overlay coloured by interaction type).

## Learning objectives

After running this notebook you will be able to:

- Explain in plain language what a **protein**, **ligand**, **binding site**, and **interaction fingerprint** are.
- Recognise the main types of non-covalent protein–ligand interactions (hydrogen bonds, hydrophobic contacts, van der Waals).
- Load a protein and a ligand in Python and compute their interaction fingerprint with [ProLIF](https://prolif.readthedocs.io/).
- Read the 3-D viewer ([py3Dmol](https://3dmol.org/)) and rotate / zoom a structure.
- Interpret the 2-D LigNetwork diagram.

## Audience

You'll get value if you are:

- A clinician or wet-lab biologist who wants to see how *in-silico* protein–ligand analysis looks before we screen new compounds.
- A data / ML person new to structural biology — this notebook introduces the vocabulary you'll see throughout the pipeline.
- A Bachelor / Master student starting on a drug-discovery thesis.

## Prerequisites

- The conda environment `aidd` is created (`conda env create -f environment.yml`), **or** you're running on Google Colab (the setup cell installs everything).
- The repo is checked out locally and the small reference data in `data/structures/` and `data/ligands/` is present.

## Runtime

Under one minute on a laptop. No GPU needed.
"""),

        markdown("""
## Key terms used in this notebook

| Term | Meaning in one line |
|---|---|
| **Protein** | A long chain of amino-acid residues folded into a 3-D shape; here, the drug target. |
| **Ligand** | A small molecule that binds to a protein. Drugs are ligands. |
| **Receptor** | Synonym for "target protein" in drug discovery contexts. |
| **Residue** | One amino acid in the protein chain. Identified by a 3-letter code (e.g. LYS, LEU) and a number. |
| **Binding site / pocket** | The cleft on the protein surface where the ligand sits. |
| **PDB file** | A plain-text format storing the 3-D coordinates of every atom in a protein / ligand. From the [Protein Data Bank](https://www.rcsb.org/). |
| **PDB ID** | A 4-character code (e.g. `4FV7`) that uniquely identifies an experimental structure in the PDB. |
| **Co-crystal** | A protein crystallised together with its ligand — gives us the experimentally observed binding pose. |
| **Pose** | A specific 3-D placement of a ligand inside a protein pocket. |
| **Interaction fingerprint (IFP)** | A binary / numeric "barcode" recording which residues interact with the ligand, and by which type of interaction. |
| **ProLIF** | The Python library we use to compute IFPs. |
| **ERK2 / MAPK1** | Extracellular signal-regulated kinase 2 — an enzyme in the MAPK signalling pathway, frequently dysregulated in cancer. |
| **Kinase** | An enzyme that adds a phosphate group to other molecules; many cancer drugs target kinases. |
| **Angstrom (Å)** | A unit of length, 10⁻¹⁰ m. Atomic bond lengths are ~1 Å, binding-site definitions usually 4–6 Å. |
"""),

        markdown("""
## Why this matters clinically

ERK2 is a kinase at the bottom of the **MAPK pathway** (RAS → RAF → MEK → ERK). The pathway is hyperactive in roughly **a third of all human cancers** — most notably melanoma, colorectal cancer, and non-small-cell lung cancer with *BRAF* or *KRAS* mutations. Drugs that block ERK2 (e.g. ulixertinib, MK-8353) are in active clinical development as second-line options when upstream MEK/BRAF inhibitors fail.

Every approved kinase inhibitor binds at — or close to — the **ATP-binding pocket** of its target. So when we compute an interaction fingerprint between ERK2 and a known inhibitor, we are mapping out the chemistry of *that* pocket. Later in the pipeline, we will use the same fingerprints as **features** for a machine-learning model that ranks new candidate compounds by how plausibly they bind.
"""),

        markdown("""
## 1. Setup

Two cells: one detects whether we're on Google Colab vs a local machine and sets the import path; the other imports the libraries we will use. The `%autoreload` magic means edits to `src/aidd/*.py` are picked up without restarting the kernel — useful when iterating.
"""),

        code("""
import sys
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules

if IS_COLAB:
    # On Colab: pull repo + install pip extras. On local conda env: skip; deps already installed.
    !pip install -q rdkit datamol "prolif>=2.0" posebusters meeko py3Dmol biopython scikit-learn xgboost lightgbm
    REPO_ROOT = Path("/content/aidd-pipeline")
    if not REPO_ROOT.exists():
        !git clone https://github.com/hvmarco/aidd-pipeline.git {REPO_ROOT}
    sys.path.insert(0, str(REPO_ROOT / "src"))
else:
    REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    sys.path.insert(0, str(REPO_ROOT / "src"))

print(f"Repo root: {REPO_ROOT}")
print(f"Running on: {'Colab' if IS_COLAB else 'local'}")
"""),

        code("""
# Auto-reload edits made to src/aidd/ without restarting the kernel.
%load_ext autoreload
%autoreload 2

import warnings
warnings.filterwarnings("ignore")  # ProLIF/RDKit/MDAnalysis are chatty

import prolif as plf

from aidd.ifp import compute_ifp, load_plf_molecule, to_wide_features
from aidd.viz import (
    show_protein_ligand,
    show_binding_site,
    show_interactions_3d,
    show_interaction_network,
)

print(f"prolif {plf.__version__}")
"""),

        markdown("""
## 2. Inputs — the ERK2 receptor and its reference ligand

### Background

We work with two files:

- **`erk2_4fv7.pdb`** — the ERK2 protein from PDB entry [4FV7](https://www.rcsb.org/structure/4FV7). The file lists every atom of the protein (and originally its co-crystal ligand) by 3-D coordinates. We've prepared it by cleaning solvent and adding hydrogens, which matters because hydrogen bonds rely on hydrogen atoms being present.
- **`erk2_4fv7_ref.pdb`** — the *ligand* from the same crystal structure, isolated into its own file. Its three-letter residue code is `E94` (a custom code from the depositors of the structure). Because it came directly from the X-ray experiment, its binding pose is *experimentally observed*, not predicted — this is what makes it a useful **reference** to validate our methods.

Throughout this notebook we will treat E94 as if it were a candidate compound and compute its interaction fingerprint. The exercise is meaningful because, in later notebooks, we will *redock* the same molecule and compare against the experimental pose to confirm our docking workflow works.
"""),

        code("""
PROTEIN = REPO_ROOT / "data" / "structures" / "erk2_4fv7.pdb"
LIGAND  = REPO_ROOT / "data" / "ligands"    / "erk2_4fv7_ref.pdb"
LIGAND_RESNAME = "E94"  # PDB three-letter code for the 4FV7 co-crystal ligand

assert PROTEIN.exists(), f"missing {PROTEIN}"
assert LIGAND.exists(),  f"missing {LIGAND}"
print(f"Protein: {PROTEIN.relative_to(REPO_ROOT)}")
print(f"Ligand:  {LIGAND.relative_to(REPO_ROOT)}")
"""),

        markdown("""
## 3. Compute the interaction fingerprint

### What an "interaction fingerprint" actually is

Imagine you put a hand on a wall. Each finger touches the wall at a different spot, and each contact has a type — knuckles vs fingertip vs side of the palm. If you wrote that down as a list of `(wall-spot, finger-type)` pairs, you'd have a **fingerprint** of how your hand touched the wall.

An **interaction fingerprint (IFP)** does the same thing for a ligand bound inside a protein pocket. For every protein residue that's close enough to the ligand, it records *which type* of non-covalent interaction is taking place. The main types we care about are:

- **Hydrogen bond (HB)** — a directional, ~3 Å contact between a hydrogen donor (e.g. an –NH) and an acceptor (e.g. a C=O). ProLIF distinguishes the donor and acceptor side: **HBDonor** when the ligand donates the H, **HBAcceptor** when the ligand accepts.
- **Hydrophobic contact** — non-polar surfaces packing together (oily groups against oily groups). No H is transferred.
- **Van der Waals contact (VdW)** — short-range attractive forces between *any* atoms that are close enough to touch. Very common; every binding-site residue typically has at least one.
- **π-stacking** — flat aromatic rings (like benzene) stacking face-to-face.
- **Salt bridge / ionic** — positive charge (e.g. lysine –NH₃⁺) meeting negative charge (e.g. carboxylate).

ProLIF detects each type using **geometric criteria** (distance and angle thresholds) on the atom coordinates in the PDB files. It is a fast, deterministic check — no machine learning involved at this stage.

### Why the IFP matters downstream

Two ligands that score similarly by raw docking energy may still differ in *which* interactions they make. A drug-discovery scientist would say: "compound A and compound B both score –9 kcal/mol, but A makes the canonical hinge hydrogen bond and B doesn't — A is more interesting". The IFP turns that intuition into a machine-readable feature. Later in the pipeline (notebook `04_score_classical`) we use IFPs as inputs to a machine-learning model that learns which interaction patterns correlate with activity.
"""),

        markdown("""
Compute the IFP. The function takes the protein PDB and the ligand PDB and returns a `pandas` DataFrame:
"""),

        code("""
ifp_df = compute_ifp(PROTEIN, LIGAND)
print(f"Shape: {ifp_df.shape}  —  one row per pose, columns = (ligand_residue, protein_residue, interaction_type)")
ifp_df
"""),

        markdown("""
### How to read this table

- **One row** because there is only one ligand pose (we loaded a single structure). When we later compute IFPs for thousands of docked poses, each pose will be one row.
- **Columns are a hierarchical (MultiIndex) label** of the form `(ligand_residue, protein_residue, interaction_type)`. E.g. `(E94.401.A, LYS112.A, VdWContact)` reads as: "the ligand E94 (residue 401, chain A) has a van der Waals contact with the protein residue lysine-112 of chain A".
- Each cell is **`True` / `False`** — was that interaction detected (`True`) or not (`False`).

The `to_wide_features` function below flattens the table into one column per *(residue, interaction)* combination, which is the format machine-learning libraries expect.
"""),

        code("""
wide = to_wide_features(ifp_df)
print(f"{wide.shape[1]} features after flattening")
wide
"""),

        markdown("""
### Interpreting the residues we hit

Look at the residue names in the columns. You should see a mix of residues from the ERK2 **ATP-binding pocket**: typically `LYS52`, `GLU31` (the catalytic K/E pair conserved across kinases), `MET106` (often the so-called **gatekeeper** residue), `ASP104`, and `LEU107`. If the model loaded correctly and the ligand is in the right pocket, these names are biologically expected; finding them is a strong sanity check.

We have not done anything sophisticated yet — we just measured geometric distances and angles on a crystal structure. The interesting part is that this same calculation, run on a *predicted* protein and a *docked* pose, gives features we can compare across thousands of candidate molecules.
"""),

        markdown("""
## 4. Look at the complex in 3-D

### Background

Before we trust any number, we should look at the structure. The next cell opens an interactive 3-D viewer in the notebook — **drag to rotate, scroll to zoom**.

Visual conventions used here (and throughout the pipeline):

- **Gold cartoon** — the protein backbone, simplified to a ribbon. Helices look like coils, sheets like flat arrows.
- **Cyan sticks** — the ligand, atom-by-atom. Carbon atoms are cyan, oxygens red, nitrogens blue.
- The viewer is rendered by [py3Dmol](https://3dmol.org/), which works the same in JupyterLab, VS Code, and Google Colab.
"""),

        code("""
view = show_protein_ligand(PROTEIN, LIGAND, ligand_resname=LIGAND_RESNAME)
view.zoomTo({"resn": LIGAND_RESNAME})
view.show()
"""),

        markdown("""
### What to look at

You should see the ligand snug inside a cleft on one face of the protein. That cleft is the **ATP-binding pocket** — under physiological conditions ATP (the natural substrate) would sit there. A successful kinase inhibitor essentially blocks ATP from getting in.

If the viewer is empty or your ligand is floating in space rather than inside a pocket, the most likely cause is that the protein and ligand files have not been aligned to the same coordinate frame. With files straight from the PDB, alignment is given for free; with predicted structures (later in the pipeline) we explicitly align.
"""),

        markdown("""
## 5. Highlight the binding-site residues

### Background

A useful trick is to colour, as sticks, every protein residue whose any atom sits within ~5 Å of any ligand atom. This is one common operational definition of the *binding site*. 5 Å is roughly the cut-off below which non-covalent interactions become possible; below 3 Å you're into hydrogen bonds, ~3–4 Å is van der Waals territory, beyond ~6 Å nothing meaningful happens between two atoms.

The residues that show up here are the ones our IFP also picked up — visual cross-check.
"""),

        code("""
view = show_protein_ligand(PROTEIN, LIGAND, ligand_resname=LIGAND_RESNAME)
show_binding_site(view, ligand_resname=LIGAND_RESNAME, radius=5.0)
view.zoomTo({"resn": LIGAND_RESNAME})
view.show()
"""),

        markdown("""
### Interpretation

The residues now drawn as gold sticks line the pocket. Hover with your mouse — most viewers show residue names on hover. Compare the residues you see here against the column labels in the IFP table above. They should overlap heavily; any residue in the IFP that *isn't* in the visual binding site is suspicious and worth investigating.
"""),

        markdown("""
## 6. The 2-D interaction network (LigNetwork)

### Background

A 3-D view is great for getting a feel, but it's hard to compare across many ligands by eye. The **LigNetwork** diagram flattens the same information onto a 2-D page: the ligand structure sits in the centre, and each interacting residue radiates outward, connected by a coloured line that encodes the interaction type. It's the diagram you'll often see in medicinal-chemistry papers when authors describe "key interactions" of a new compound.
"""),

        code("""
# Re-build the fingerprint object (we need it, not just the DataFrame, for LigNetwork)
protein_plf = load_plf_molecule(PROTEIN)
ligand_plf  = load_plf_molecule(LIGAND)

fp = plf.Fingerprint()
fp.run_from_iterable([ligand_plf], protein_plf, progress=False)

show_interaction_network(fp, ligand_plf)
"""),

        markdown("""
### How to read the diagram

- The **2-D skeleton** at the centre is the ligand. Each line is a chemical bond; each labelled letter is an atom (C carbon, N nitrogen, O oxygen…).
- Each **labelled blob** around the edge is a protein residue.
- The **coloured line** between an atom of the ligand and a residue tells you both *which atom* is involved and *what kind* of interaction it is (the legend lives at the top of the figure).
- Thicker / heavier lines indicate interactions seen in more poses when you have multiple poses. With a single pose (our case), all lines look the same.

This is the diagram you would put in a paper figure to describe a new inhibitor's binding mode.
"""),

        markdown("""
## 7. 3-D view with interactions overlaid

### Background

The final figure layers the IFP back onto the 3-D viewer: for every detected interaction, a dashed cylinder is drawn between the ligand atom and the protein-residue atom that are interacting, coloured by interaction type. This is the figure to compare directly with the canonical figure in the original Leiden Week-3 docking notebook.

By default we **hide van der Waals contacts** because almost every binding-site residue has at least one VdW contact and the view becomes a wire cage. The chemically discriminating interactions (hydrogen bonds, hydrophobic packing, π-stacking) remain. Pass `skip_types=()` if you want VdW back in.
"""),

        code("""
view = show_protein_ligand(PROTEIN, LIGAND, ligand_resname=LIGAND_RESNAME)
show_interactions_3d(view, fp, protein_plf, ligand_plf, pose_index=0)
view.zoomTo({"resn": LIGAND_RESNAME})
view.show()
"""),

        markdown("""
### Interpretation

Colour key (defined in `src/aidd/viz.py`):

- **lime** — hydrophobic
- **red** — hydrogen-bond *acceptor* (ligand accepts H from protein)
- **blue** — hydrogen-bond *donor* (ligand donates H to protein)
- **purple** — π-stacking
- **orange** — π-cation
- **grey** — anything not in the map above

Rotate the view so you can see each cylinder. For a kinase inhibitor the most diagnostic interaction is usually a **hydrogen bond to the hinge region** — a short stretch of residues at the back of the ATP pocket where ATP itself H-bonds the protein backbone. If our ligand makes that contact, it is engaging the pocket the right way.
"""),

        markdown("""
## Recap

### Biomedical takeaway

We loaded a co-crystal of the cancer-relevant kinase ERK2 with a small-molecule inhibitor, identified the residues that line the ATP-binding pocket, and characterised the chemistry of how the inhibitor sits there. The list of interactions (one ATP-pocket lysine in a hydrogen bond, several hydrophobic contacts deeper in the pocket, etc.) is the kind of fingerprint a medicinal chemist looks for when designing the next generation of kinase inhibitors. The same procedure works for any protein–ligand complex.

### Technical takeaway

Two short Python modules (`aidd.ifp`, `aidd.viz`) wrap the standard ProLIF + py3Dmol idioms into three composable functions: read a complex, get the IFP as a `pandas` DataFrame, and visualise it. The DataFrame is in the shape that downstream machine-learning models (random forests, gradient boosting, etc.) can consume directly — that's the bridge into notebook `04_score_classical`.

### What's next in the pipeline

- **`02_prepare_ligands.ipynb`** — take a SMILES library and turn it into 3-D, drug-like, PAINS-clean molecules ready for docking.
- **`01_fold_target.ipynb`** *(Colab)* — predict a 3-D structure from a protein sequence when no PDB is available, using ColabFold.
- **`03_dock_gnina.ipynb`** — actually dock a library of compounds into the receptor.
- **`04_score_classical.ipynb`** — train a machine-learning rescorer on IFPs.

### Further reading

- Bouysset & Fiorucci, *J. Cheminform.* (2021), **13**, 72 — *ProLIF: a library to encode molecular interactions as fingerprints.* [doi:10.1186/s13321-021-00548-6](https://doi.org/10.1186/s13321-021-00548-6)
- Roskoski, *Pharmacol. Res.* (2024) — yearly update on FDA-approved protein-kinase inhibitors. Good context on which kinases (incl. ERK1/2) are being drugged.
- Volkamer-Lab **TeachOpenCADD** — open educational notebooks on every step of computer-aided drug design, written in the same style. [projects.volkamerlab.org/teachopencadd](https://projects.volkamerlab.org/teachopencadd/)
"""),
    )
    save(nb, NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
