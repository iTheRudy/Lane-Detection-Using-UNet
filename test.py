import torch
import numpy as np
import cv2
from torchvision import transforms
from model import UNet

def iou(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter / union).item()

def dice(pred, target):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum())).item()

def pixel_accuracy(pred, target):
    pred = (pred > 0.5).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()

def run_testing(model_path, test_images_path, test_masks_path):

    images = np.load(test_images_path)
    masks = np.load(test_masks_path)

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    iou_scores = []
    dice_scores = []
    acc_scores = []

    outputs = []

    for i in range(len(images)):
        img = images[i]
        mask = masks[i]

        inp = transform(cv2.resize(img, (200, 64))).unsqueeze(0).to(device)
        mask_tensor = transform(cv2.resize(mask, (200, 64))).to(device)

        with torch.no_grad():
            pred = model(inp)[0][0]

        iou_scores.append(iou(pred, mask_tensor))
        dice_scores.append(dice(pred, mask_tensor))
        acc_scores.append(pixel_accuracy(pred, mask_tensor))

        if i < 5:
            pred_img = (pred.cpu().numpy() * 255).astype("uint8")
            pred_path = f"prediction_{i}.png"
            mask_path = f"mask_{i}.png"
            img_path = f"image_{i}.png"

            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(pred_path, pred_img)

            outputs.extend([img_path, mask_path, pred_path])

    results = {
        "test_iou": sum(iou_scores) / len(iou_scores),
        "test_dice": sum(dice_scores) / len(dice_scores),
        "test_accuracy": sum(acc_scores) / len(acc_scores),
    }

    return results, outputs
