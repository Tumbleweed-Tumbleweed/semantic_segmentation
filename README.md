# Semantic Segmentation CNN (PyTorch)

This repository contains a simple semantic segmentation project built from the ground up. It is designed to run on any personal computer with a modern NVIDIA GPU. Semantic segmentation is a computer vision task whose goal is to label every pixel in an image based on what it represents, such as animals, people, or objects. This model was specifically implemented, trained, and tested on the PASCAL VOC 2012 dataset, and while it can be used with other datasets, its performance outside of PASCAL VOC has not been tested. The model itself is a convolutional neural network and has two implementations that can be selected from: a default version and a similar version that also includes skip connections. In addition to the model, this project also implements a custom dataloader that processess the images for model use.

## Repository Structure
```
scripts/train.py          Training entry point
src/dataset.py            PASCAL dataset loader
src/semantic_cnn.py       CNN model definitions
PASCAL_Segmentation/      Dataset directory
segmentation_output/      Output directory
```

## Dataset
Place the dataset at the repository root in the following format:
```
PASCAL_Segmentation/
  Images/
    *.jpg
  Annotations/
    *.png
```

## Run
From the repository root:
```bash
python scripts/train.py
```

## Outputs
After training, the following files are saved to:
```
segmentation_output/
  sample_train_prediction.png
  sample_val_prediction.png
  training_curves.png
```

## Notes
- `semantic_model_type = 0` uses the baseline encoder–decoder model  
- `semantic_model_type = 1` uses the skip-connection model  

The skip-connection model is recommended, as it generally runs faster and produces better results.

Training parameters can be adjusted directly in `scripts/train.py`.

