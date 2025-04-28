import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from utils.vit_linear import ViTLinearClassifier
from utils.lfw_pair_dataset import LFWDatasetPair


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def load_model(model_path, device, num_classes=2000):
    model = ViTLinearClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.head = nn.Identity()
    model.to(device)
    model.eval()
    return model


def evaluate(model, dataloader, device, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader, desc="Extracting & comparing"):
            img1, img2 = img1.to(device), img2.to(device)
            labels = torch.tensor(labels).to(device)

            feat1 = model(img1)
            feat2 = model(img2)

            sims = F.cosine_similarity(feat1, feat2)
            all_scores.append(sims.cpu())
            all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()

    print("\n Verification Accuracy:")
    for t in thresholds:
        preds = (all_scores > t).astype(int)
        acc = (preds == all_labels).sum() / len(all_labels)
        print(f"Threshold {t:.2f} → Accuracy: {acc:.4f}")

    # === ROC curve & AUC ===
    fpr, tpr, thres = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    print(f"\n AUC = {roc_auc:.4f}")
    
    target_fpr = 0.01
    idx = (fpr >= target_fpr).nonzero()[0]
    if len(idx) > 0:
        tpr_at_fpr = tpr[idx[0]]
        print(f"TPR @ FPR = 0.01: {tpr_at_fpr:.4f}")
    else:
        print("FPR=0.01 没有达到，可能验证集太小或者模型区分度不够")

    # Best threshold
    best_idx = np.argmax(tpr - fpr)
    best_thresh = thres[best_idx]
    print(
        f"Best threshold = {best_thresh:.4f}, TPR = {tpr[best_idx]:.4f}, FPR = {fpr[best_idx]:.4f}"
    )

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LFW Face Verification ROC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve.png")
    print("ROC curve saved as roc_curve.png")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform()
    dataset = LFWDatasetPair(root_dir=args.lfw_dir, transform=transform, occlusion='mouth')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model_path, device)
    evaluate(model, dataloader, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="models/vit_occluded.pth"
    )
    parser.add_argument("--lfw_dir", type=str, default="data/LFW")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)
