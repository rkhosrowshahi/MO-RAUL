# MO-RAUL: Multi-Objective RAUL

This directory contains the Multi-Objective RAUL recipe: machine unlearning framed as two-objective optimization over the same network—minimize loss on retained data while actively forgetting via a second objective on the forget set. Gradients from both objectives are combined with Jacobian descent ([torchjd](https://github.com/TorchJD/torchjd)) aggregators (for example UPGrad, MGDA, or linear scalarization), rather than hand-tuned loss weights alone.

The implementation lives in `src/unlearn/mojd/` and is intended to be used with the parent MOMU codebase at the repository root (training scripts, datasets, checkpoints). See the main [README.md](../README.md). This folder is a focused fork slice: updated library code plus YAML configs under `configs/`, not a full standalone repo (no duplicate `scripts/` or `requirements.txt`). The overall codebase lineage follows [Unlearning Saliency](https://github.com/OPTML-Group/Unlearn-Saliency).

---

## Objectives and registered methods

| Config / code name | `unlearn:` value in YAML | Forget-side objective (high level) |
|--------------------|--------------------------|------------------------------------|
| MO-CE + NegCE | `MO_JD_CE_CE` | Retain CE vs. forget **negative CE** (push wrong on forget) |
| MO-CE + HeldDist | `MO_JD_CE_HeldDist` | Retain CE vs. **class-wise KL** aligning forget predictions to validation (held) class marginals |
| MO-CE + UniDist | `MO_JD_CE_UniDist` | Retain CE vs. **cross-entropy to uniform** (high-entropy forget outputs) |
| MO-CE + UniDist (calibrated) | `MO_JD_CE_UniDist_Calib` | UniDist variant with calibration-related handling (see `CE_UniDist_Calib.py`) |

Aggregators are selected with `mo_aggregator` in the config (for example `upgrad`, `mgda`, and `mean`—used by the **`LS_*` config folders** as the equal-weight average of per-objective gradients—plus other [torchjd](https://github.com/TorchJD/torchjd) aggregators implemented in the training loop).

---

## Layout in this folder

```
MO-RAUL/
├── configs/cifar10/
│   ├── random_10percent/raul/...
│   └── random_50percent/raul/...
└── src/
    ├── arg_parser.py          # if present: overrides vs parent
    └── unlearn/
        ├── __init__.py        # registers MO-RAUL MOJD methods
        └── mojd/              # CE_NegCE, CE_HeldDist, CE_UniDist, CE_UniDist_Calib, utils
```

Example config path pattern:

`configs/cifar10/{random_10percent|random_50percent}/raul/{MO_CE_HeldDist|MO_CE_UniDist}/.../run_0N.yaml`

Each method directory typically includes runs for **SGD / AdamW** and aggregators such as **LS** (`mo_aggregator: mean`), **UPGrad**, and **MGDA** (exact set may vary by experiment).

---

## Using MO-RAUL with the main MOMU tree

1. **Merge source**: Copy `MO-RAUL/src/unlearn/` (and `MO-RAUL/src/arg_parser.py` if you rely on its flags) into the parent project’s `src/`, replacing the matching modules so `get_unlearn_method` exposes `MO_JD_CE_HeldDist`, `MO_JD_CE_UniDist`, and `MO_JD_CE_UniDist_Calib` as in `MO-RAUL/src/unlearn/__init__.py`.

2. **Merge configs**: Copy `MO-RAUL/configs/` into the parent `configs/` so paths like `./checkpoints/...` and `./logs/...` match your machine.

3. **Run from the parent project root** (where `scripts/` lives), same as upstream MOMU:

   ```bash
   pip install -r requirements.txt
   python scripts/main_forget.py --config path/to/MO-RAUL/configs/cifar10/random_10percent/raul/MO_CE_UniDist/MO_CE_UniDist_UPGrad_AdamW/run_01.yaml --gpu 0
   ```

   To sweep seeds:

   ```bash
   python scripts/run_config_dir.py path/to/MO-RAUL/configs/cifar10/random_10percent/raul/MO_CE_UniDist/MO_CE_UniDist_UPGrad_AdamW --gpu 0
   ```

Ensure **pretrained** and **Retrain** baselines exist at the paths referenced in each YAML (`model_path`, `retrain_path`), as in the main MOMU README.

---

## Dependencies

Same as parent MOMU: **PyTorch**, **torchvision**, [torchjd](https://github.com/TorchJD/torchjd), **wandb**, and related packages from the root `requirements.txt`. MO-RAUL adds no separate dependency file.

---

## Citation

If you use this multi-objective RAUL formulation, please cite in your paper.
