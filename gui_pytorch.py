import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn



class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.e1 = CBR(3, 32)
        self.e2 = CBR(32, 64)
        self.e3 = CBR(64, 128)

        self.pool = nn.MaxPool2d(2, 2)

        self.d3 = CBR(128, 64)
        self.d2 = CBR(64, 32)
        self.d1 = nn.Conv2d(32, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        e1 = self.e1(x)
        p1 = self.pool(e1)

        e2 = self.e2(p1)
        p2 = self.pool(e2)

        e3 = self.e3(p2)

        d3 = self.up(e3)
        d3 = self.d3(d3)

        d2 = self.up(d3)
        d2 = self.d2(d2)

        out = torch.sigmoid(self.d1(d2))
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("lane_unet.pth", map_location=device))
model.eval()

print("Model loaded successfully.")



class LaneGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lane Detection - UNet")
        self.root.geometry("900x650")
        self.root.configure(bg="#222222")

        self.upload_btn = tk.Button(
            root,
            text="Upload Image",
            command=self.upload_image,
            width=20,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 14)
        )
        self.upload_btn.pack(pady=20)

        self.original_label = tk.Label(root, bg="#222222")
        self.original_label.pack()

        self.mask_label = tk.Label(root, bg="#222222")
        self.mask_label.pack()

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if path == "":
            return

        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Preprocess
        resized = cv2.resize(img_rgb, (200, 66))
        tensor = torch.tensor(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            pred = model(tensor).cpu().numpy()[0][0]

        mask = (pred * 255).astype(np.uint8)

        # Resize for display
        disp_img = cv2.resize(img_rgb, (500, 300))
        disp_mask = cv2.resize(mask, (500, 300))

        # Tkinter images
        img_tk = ImageTk.PhotoImage(Image.fromarray(disp_img))
        mask_tk = ImageTk.PhotoImage(Image.fromarray(disp_mask))

        self.original_label.config(image=img_tk)
        self.original_label.image = img_tk

        self.mask_label.config(image=mask_tk)
        self.mask_label.image = mask_tk



root = tk.Tk()
app = LaneGUI(root)
root.mainloop()
