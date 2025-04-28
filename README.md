# Vision Transformer for Face Recognition under Occlusion

This project implements a **Vision Transformer (ViT)** model for face recognition, with a focus on improving recognition accuracy under various occlusion conditions.

## Project Structure

```
Occlusion-ViT/
├── data/                              # Dataset files (CelebA, LFW, etc.)
│   ├── celeba/                        # CelebA dataset
│   └── LFW/                           # LFW dataset
├── models/                            # Model definitions and saved weights
│   ├── vit_occluded.pth               # Trained ViT model on CelebA dataset
├── utils/                             # Source code for model training and evaluation
│   ├── celeba_dataset.py              # Dataset loading logic for CelebADataset
│   ├── vit_linear.py                  # Model loading logic
│   └── lfw_pair_dataset.py            # Dataset loading logic for LFW image pairs
│
└── README.md
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Balencicash/Occlusion-ViT.git
   cd Occlusion-ViT
   ```

2. Download the **CelebA** and **LFW** datasets:

   - [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   - [LFW](http://vis-www.cs.umass.edu/lfw/)

## Training

To train the ViT model on the occluded **CelebA** dataset, run:

```bash
python -u train_vit_linear_occluded.py \
        --data_dir data/celeba/img_align_celeba \
        --label_path data/celeba/identity_CelebA.txt \
        --include_occlusions \
        --resume models/vit_occluded_best.pth \
```

## Validation

Once trained, validate the model on the **LFW** dataset:

```bash
python -u eval_vit_occluded.py
```

## Result

Pretrained weight(b16_224 version from **timm** library) performance:

![](/img/roc_val_pretrained_eye.png)

Improved performance:

![](/img/roc_val_occluded_eye.png)

Both of the performances are under the eyes covered situation on the LFW verification task. You can check the **img** directory for more performance plots.

## License

MIT License - see [LICENSE](LICENSE) for details.
