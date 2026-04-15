# Mukara: A Deep Learning Alternative to the Four-Step Travel Demand Model

![High-resolution PDF version of the framework figure](manuscript/framework.pdf)

This repository contains the TensorFlow + DGL implementation of **Mukara**, a deep learning framework for interurban highway traffic volume prediction using external socioeconomic and network features.

## Paper
- **Expected publication date**: 16 April 2026
- **Journal page**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0345576
- **DOI**: https://doi.org/10.1371/journal.pone.0345576

## Repository Structure
- `train.py`: Main training entry point.
- `config.py`: Active experiment configuration.
- `config_template.py`: Template configuration for new experiments.
- `model/`: Core model and data-loading code.
  - `model_full.py`: Full Mukara model.
  - `model_no_edge_features.py`: Ablation without edge features.
  - `model_only_edge_features.py`: Ablation with only edge features.
  - `dataloader.py`: Data loading and preprocessing helpers.
  - `utils.py`: Metrics and utility functions.
- `eval/`: Evaluation scripts and notebooks.
  - `evaluate_log.py`: Parse and summarize training logs.
- `inference.ipynb`: Inference and prediction workflow.
- `train_batch.ipynb`: Batch training workflow.
- `data/`: Data processing scripts and assets.
- `manuscript/`: Repository-kept manuscript image asset (`mukara.png`).

## Environment
Tested environment used in the study:
- Operating system: Windows 11
- Python: 3.9.18
- GPU: NVIDIA RTX 4090

Key dependencies:
- `tensorflow==2.10.1`
- `dgl==1.1.2+cu118`
- `numpy==1.25.2`
- `pandas==2.0.3`
- `cudatoolkit==11.2.2`
- `cudnn==8.1.0.77`

## Data
All datasets should be downloaded directly from their official sources.

This repository provides preprocessing and modeling code only. Place processed datasets under `data/` following the relative paths defined in `config.py` (`PATH` block).

## Training
1. Adjust `config.py` as needed.
2. Run training:

```bash
python train.py
```

Model weights are saved to `param/`, and logs are written to `eval/logs/`.

All Jupyter notebooks in this repository are versioned with cleared outputs to keep diffs lightweight and review-friendly.

## Citation
If you use this repository, please cite:

```bibtex
@article{li2026mukara,
  author  = {Li, Yue and Chen, Shujuan and Jin, Ying},
  title   = {Mukara: a deep learning alternative to the four-step travel demand model with a case study on interurban highway traffic prediction in the UK},
  journal = {PLOS ONE},
  year    = {2026},
  doi     = {10.1371/journal.pone.0345576},
  url     = {https://doi.org/10.1371/journal.pone.0345576},
  note    = {Expected publication date: 16 April 2026}
}
```

## License
This project is licensed under the MIT License. See `LICENSE` for details.
