import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2
from model import UNet

TRAIN_P = "full_CNN_train.p"
LABELS_P = "full_CNN_labels.p"

class LaneDataset(Dataset):
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = cv2.resize(self.x[i], (200, 64))
        mask = cv2.resize(self.y[i], (200, 64))
        return self.t(img), self.t(mask)

def run_training():
    images = np.array(pickle.load(open(TRAIN_P, "rb")))
    labels = np.array(pickle.load(open(LABELS_P, "rb")))

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.36, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5555, random_state=42)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    train_loader = DataLoader(LaneDataset(X_train, y_train, transform), batch_size=16, shuffle=True)
    val_loader = DataLoader(LaneDataset(X_val, y_val, transform), batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
        model.train()
        total = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1} Loss: {total/len(train_loader):.4f}")

    model_path = "lane_unet.pth"
    torch.save(model.state_dict(), model_path)

    np.save("test_images.npy", X_test)
    np.save("test_masks.npy", y_test)

    return model_path, "test_images.npy", "test_masks.npy"
