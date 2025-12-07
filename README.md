# PyTorch Lane Detection (U-Net)

This project implements lane detection using a U-Net model.

## Features
- U-Net lane segmentation (PyTorch)

- GUI for testing

## Install
```
pip install -r requirements.txt
```

## Train
```
python train.py --train_p full_CNN_train.p --labels_p full_CNN_labels.p --epochs 15 --save_path lane_unet.pth
```

## GUI
```
python gui_pytorch.py --model lane_unet.pth
```

## DataSets
 - Training Images: https://www.dropbox.com/scl/fi/2pjg0rnfq1o107evi0qu6/full_CNN_train.p?rlkey=xtf9bavh3vy9iy8ggnfvl87ps&st=zm0d1pa7&dl=0

 - Training Labels: https://www.dropbox.com/scl/fi/2pjg0rnfq1o107evi0qu6/full_CNN_train.p?rlkey=xtf9bavh3vy9iy8ggnfvl87ps&st=qlcaufrs&dl=0


As the Presentation Video is too large to fit the 100mb limit I will be Uploading it to github as a compressed file.

