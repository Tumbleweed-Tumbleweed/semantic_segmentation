# train.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
import torch
import torch.nn as nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.dataset import PascalDataSet
from src.semantic_cnn import SemanticEncoderCNN, skipConnectionCNN

#-------------------------#
#   function declaration  #
#-------------------------#

# based on homework 6 IoU mertirc function. has some slight tweaks. used for model evaluation
def iou(preds, targets, ignore_index=255):
    # this version computes IoU both with and without the background class as I was having issues where I thought I had a good model but it was only good predicting the background
    C = preds.size(1)
    ious = np.zeros(C)
    counts = np.zeros(C)
    for b in range(preds.size(0)):
        pred = torch.argmax(preds[b], dim=0)
        valid = targets[b] != ignore_index
        for c in range(C):
            gt_c = (targets[b] == c) & valid
            if gt_c.any():
                pred_c = (pred == c) & valid
                inter = (pred_c & gt_c).sum()
                union = (pred_c | gt_c).sum()
                ious[c] += (inter / union).item()
                counts[c] += 1
    return ious, counts

# model training loop - may need changes
def training_loop(model, criterion, optimizer, n_epochs, train_loader, val_loader, device,
                  ignore_index=255, num_classes=21):
    loss_values, train_ious, val_ious = [], [], []

    for n in tqdm(range(n_epochs)):
        # ---- TRAIN ----
        model.train()
        epoch_loss = 0.0
        epoch_ious = np.zeros(num_classes)
        epoch_counts = np.zeros(num_classes)

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)                 # (B,C,H,W)
            loss = criterion(preds, y_batch)       # (B,H,W)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            ious, counts = iou(preds, y_batch, ignore_index=ignore_index)
            epoch_ious += ious
            epoch_counts += counts

        loss_values.append(epoch_loss / max(1, len(train_loader)))
        train_fg = epoch_ious[1:] / np.maximum(epoch_counts[1:], 1)
        train_all = epoch_ious / np.maximum(epoch_counts, 1)
        train_ious.append({"foreground": np.mean(train_fg),"overall": np.mean(train_all)})

        # ---- VALID ----
        model.eval()
        epoch_val_ious = np.zeros(num_classes)
        epoch_val_counts = np.zeros(num_classes)

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch)
                ious, counts = iou(preds, y_batch, ignore_index=ignore_index)
                epoch_val_ious += ious
                epoch_val_counts += counts

        val_fg = epoch_val_ious[1:] / np.maximum(epoch_val_counts[1:], 1)
        val_all = epoch_val_ious / np.maximum(epoch_val_counts, 1)

        val_ious.append({"foreground": np.mean(val_fg),"overall": np.mean(val_all)})

    return model, loss_values, train_ious, val_ious

# gives the output more color variety for better clarity, also sets background to black
def get_voc_colormap(num_classes=21):
    colors = np.zeros((num_classes, 3))
    colors[0] = [0.0, 0.0, 0.0]

    base_colors = [
        (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1),
        (0,1,1), (1,0.5,0), (0.5,1,0), (0,0.5,1),
        (0.5,0,1), (1,0,0.5), (0,1,0.5),
        (0.7,0.7,0), (0.7,0,0.7), (0,0.7,0.7),
        (1,0.3,0.3), (0.3,1,0.3), (0.3,0.3,1),
        (0.9,0.6,0.2), (0.2,0.6,0.9),
    ]

    for i in range(1, min(num_classes, len(base_colors)+1)):
        colors[i] = base_colors[i-1]

    return mcolors.ListedColormap(colors)

# function to give visual feedback for model
def save_one_sample(model, loader, device, out_path):
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = next(iter(loader))          # one batch
        x_batch = x_batch.to(device)
        logits = model(x_batch).cpu()                  # (B,C,H,W)

    x0 = x_batch[0].cpu()                              # (3,H,W)
    y0 = y_batch[0].cpu()                              # (H,W)
    pred0 = torch.argmax(logits[0], dim=0).cpu()       # (H,W)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    x_np = x0.permute(1, 2, 0).numpy()
    x_np = (x_np * std) + mean
    x_np = np.clip(x_np, 0, 1)

    plt.figure(figsize=(12, 4))

    cmap = get_voc_colormap(21)

    plt.subplot(131)
    plt.title("Input")
    plt.imshow(x_np)
    plt.axis("off")

    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(y0.numpy(), cmap=cmap, vmin=0, vmax=20)
    plt.axis("off")

    plt.subplot(133)
    plt.title("Prediction")
    plt.imshow(pred0.numpy(), cmap=cmap, vmin=0, vmax=20)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

#-------------------------#
#       main program      #
#-------------------------#

# tweakable input parameters
N = 5
N_train = int(N * 0.6)        # data will be split between training and validaiton
N_val = N - N_train
batch_size = 2
image_size = 128               # image size tweaking
learning_rate = 1e-4
set_weight_decay = 1e-4
n_epochs = 250
semantic_model_type = 1         # 0 indicates first CNN Model, 2 indicates the split connections model

# dataset creation
image_folder = REPO_ROOT / "PASCAL_Segmentation" / "Images"
all_names = sorted([n for n in os.listdir(image_folder) if n.endswith(".jpg")])[:N]
all_names = np.random.permutation(all_names)[:N]

train_names = all_names[:N_train]
val_names   = all_names[N_train:N_train + N_val]

train_dataset = PascalDataSet(root="PASCAL_Segmentation", names=train_names, image_size=image_size, training=True)
val_dataset   = PascalDataSet(root="PASCAL_Segmentation", names=val_names,   image_size=image_size, training=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# load gpu, select model to run and begin training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if semantic_model_type == 0:
    model = SemanticEncoderCNN(number_classes=21).to(device)
elif semantic_model_type == 1:
    model = skipConnectionCNN(number_classes=21).to(device)


weights = torch.ones(21, device=device)
weights[0] = 0.1            # tweakable
criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=set_weight_decay)           # im using Adam becuase my math major friend used Adam for his Symplectic mapping project and it was the only part of his entire project I could understand
model, loss_values, train_ious, val_ious = training_loop(model, criterion, optimizer, n_epochs, train_loader, val_loader, device, ignore_index=255, num_classes=21)

#-------------------------#
#     output plotting     #
#-------------------------#

#output data location
output_folder = REPO_ROOT / "output"
os.makedirs(output_folder, exist_ok=True)

save_one_sample(model=model,loader=val_loader, device=device,out_path=str(output_folder / "sample_val_prediction.png"))
save_one_sample(model=model,loader=train_loader, device=device,out_path=str(output_folder / "sample_train_prediction.png"))

plt.figure(figsize=(15, 6))

plt.subplot(131)
plt.semilogy(loss_values)
plt.grid(True)
plt.title("Loss values")
plt.xlabel("Epoch")

plt.subplot(132)
plt.plot([t["foreground"] for t in train_ious], label="NO BG")
plt.plot([t["overall"] for t in train_ious], label="All")
plt.legend()
plt.title("Training mIoU")

plt.subplot(133)
plt.plot([v["foreground"] for v in val_ious], label="NO BG")
plt.plot([v["overall"] for v in val_ious], label="All")
plt.legend()
plt.title("Validation mIoU")

plt.tight_layout()
plt.savefig(str(output_folder / "training_curves.png"))
plt.close()

final_loss = loss_values[-1]

# I was losing tracj of what tests ran what parameters so this makes it clear
print("\n===== FINAL RESULTS =====")
print(f"Final Training Loss: {final_loss:.4f}")
print(f"Final Train mIoU (FG): {train_ious[-1]['foreground']:.4f}")
print(f"Final Train mIoU (All): {train_ious[-1]['overall']:.4f}")
print(f"Final Val mIoU (FG): {val_ious[-1]['foreground']:.4f}")
print(f"Final Val mIoU (All): {val_ious[-1]['overall']:.4f}")
print("=========================\n")

print("\n===== EXPERIMENT CONFIG =====")
print(f"N (total images): {N}")
print(f"N_train: {N_train}")
print(f"N_val: {N_val}")
print(f"Batch size: {batch_size}")
print(f"Image size: {image_size} x {image_size}")
print(f"Learning rate: {learning_rate}")
print(f"Weight decay: {set_weight_decay}")
print(f"Epochs: {n_epochs}")
print(f"Model type: {semantic_model_type}")
print("================================\n")

print("===== DATASET CHECK =====")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print("=========================\n")
