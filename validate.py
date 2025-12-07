import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from model import UNet

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

def iou(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / union).item()

def run_validation(model_path, test_images_path, test_masks_path):
    images = np.load(test_images_path)
    masks = np.load(test_masks_path)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    loader = DataLoader(LaneDataset(images, masks, transform), batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scores = []
    for imgs, m in loader:
        imgs, m = imgs.to(device), m.to(device)
        with torch.no_grad():
            preds = model(imgs)
            for p, t in zip(preds, m):
                scores.append(iou(p, t))

    return sum(scores) / len(scores)
