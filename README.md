# Crossing-aware auxiliary supervision for 12-lead ECG image digitisation

This is the code for the ExplAI-2026 paper by V. Petryna and V. Lytvynenko. It covers the experiments in the paper: building the strip cache, training the base reader and the A/C ablation models, scoring with the record-clustered bootstrap, the PTB-XL checks, and the figures.

## Setup

```bash
uv sync                      # or: pip install -e .
```

Data, placed in `data/`:

- `physionet-ecg-image-digitization.zip`: the training data of the Kaggle competition "PhysioNet – Digitization of ECG Images". Download it from Kaggle under the competition rules.
- `ptb-xl-1.0.3.zip`: PTB-XL 1.0.3 from PhysioNet (https://doi.org/10.13026/kfzx-aw45, CC BY 4.0). Only the ablation cache and the PTB-XL checks need it.

Third-party rectifier: the page rectifier is not part of this repository. Download the code and weights by Kaggle user hengck23 into `third_party/hengck23/`:

- `stage0_common.py`, `stage0_model.py`, `stage1_common.py` and `stage1_model.py`, from the Kaggle dataset `hengck23/hengck23-demo-submit-physionet`
- `weight/stage0-last.checkpoint.pth` and `weight/stage1-last.checkpoint.pth`, from the same dataset (version of 14 November 2025)

```bash
kaggle datasets download hengck23/hengck23-demo-submit-physionet -p third_party/hengck23 --unzip
```

## Development split

`crossecg.prepare.split_records` splits the 977 records with seed 2026: 10% of the records (98) are the development set. `data/development_records.txt` lists them.

## Steps

```bash
# 1. PTB-XL matches (500 Hz labels for the ablation cache) and the patient-overlap check
python scripts/ptbxl_checks.py --source data/physionet-ecg-image-digitization.zip --ptbxl data/ptb-xl-1.0.3.zip

# 2. Strip caches (rectify every page, cut 4 strips, register labels)
python -m crossecg.prepare --output-dir outputs/cache_base --train-kinds-per-record 3
python -m crossecg.prepare --output-dir outputs/cache_ablation --train-kinds-per-record 9 \
    --height 480 --vertical-scale 1.0 --label-scale 4 \
    --ptbxl data/ptb-xl-1.0.3.zip --ptbxl-matches outputs/ptbxl_matches.csv

# 3. Training (one GPU, fixed wall-clock budget)
for c in base base_seed2 base_long ablA_seed1 ablC_seed1 ablA_seed2 ablC_seed2; do
    python -m crossecg.train --config configs/$c.json
done

# 4. Development-set predictions
python scripts/dump_predictions.py --checkpoint outputs/runs/ablC_seed1/checkpoints/last.pt \
    --cache-dir outputs/cache_ablation --source data/physionet-ecg-image-digitization.zip \
    --output outputs/predictions/ablC_seed1_val_pred.pkl
# ... the same for every run (base runs use outputs/cache_base)

# 5. Results
python scripts/ensemble.py outputs/predictions/base_seed2_val_pred.pkl outputs/predictions/base_long_val_pred.pkl
python scripts/score_ablation.py --pairs \
    outputs/predictions/ablA_seed1_val_pred.pkl:outputs/predictions/ablC_seed1_val_pred.pkl \
    outputs/predictions/ablA_seed2_val_pred.pkl:outputs/predictions/ablC_seed2_val_pred.pkl
python scripts/paper_figures.py      # writes outputs/figures
```

Training times are wall-clock budgets on one NVIDIA RTX PRO 6000, so the number of updates, and therefore the scores, depends on the hardware. Figure 5 uses the in-training validation history (`history.json`) of the seed-2 runs.

## Layout

- `crossecg/layout.py`: page and strip geometry, lead windows, per-group time offsets
- `crossecg/rectify.py`: wrapper around the third-party rectifier
- `crossecg/labels.py`: rendering of reference signals into strip labels, ink registration
- `crossecg/prepare.py`: cache builder and record split
- `crossecg/dataset.py`: training windows, augmentation, crossing targets
- `crossecg/model.py`: U-Net with ConvNeXt encoder; optional crossing head (training only)
- `crossecg/train.py`: losses and the training loop
- `crossecg/signals.py`: least-squares column-to-sample inversion, lead II fusion, limb-lead projection
- `crossecg/predict.py`: inference and scoring
- `crossecg/metric.py`: the competition SNR metric

## License

MIT (see `LICENSE`). The third-party rectifier and the datasets keep their own terms.
