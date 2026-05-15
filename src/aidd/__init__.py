"""aidd — in-silico screening pipeline package.

Module layout (planned):
- structures: PDB I/O, alignment, distogram QC
- ligands: SMILES standardisation, 3D embedding, drug-likeness filtering
- docking: gnina + Vina wrappers (CPU), Boltz-2 wrapper (Colab/GPU)
- ifp: ProLIF wrappers, interaction-fingerprint feature extraction
- scoring: per-target ML rescorer (sklearn / XGBoost)
- viz: py3Dmol helpers
- io: path conventions and per-stage cache management
- variants: AlphaMissense + gnomAD per-variant priors (notebook 07)
"""

__version__ = "0.0.0"
