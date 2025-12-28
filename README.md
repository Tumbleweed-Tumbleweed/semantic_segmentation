# Semantic Segmentation CNN (PyTorch)

This project implements a custom encoder–decoder CNN for semantic segmentation, with an optional skip-connection variant. The model is trained on a PASCAL-style dataset and saves prediction samples and training curves.

Repository Structure
--------------------
scripts/train.py        training entry point
src/dataset.py          PascalDataSet
src/semantic_cnn.py     CNN models
PASCAL_Segmentation     dataset folder (not tracked by git)
segmentaion_output      outputs (not tracked by git)

Requirements
------------
Install dependencies:
pip install -r requirements.txt

Dataset
-------
Place the dataset in the following format at the repository root:

PASCAL_Segmentation
  Images
    *.jpg
  Annotations
    *.png

Run
---
From the repository root:
python scripts/train.py

Outputs
-------
After training, the following files are saved to:

segmentaion_output
  sample_train_prediction.png
  sample_val_prediction.png
  training_curves.png

Notes
-----
semantic_model_type = 0 uses the baseline encoder–decoder model
semantic_model_type = 1 uses the skip-connection model
Training parameters can be adjusted directly in scripts/train.py
