import os
import timm
import time
import json
import torch
import socket
import random
import argparse
import torchvision
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from utils.celeba_dataset import CelebADataset
from utils.vit_linear import ViTLinearClassifier
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split


# set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# arguments
parser = argparse.ArgumentParser(description='Train ViT Linear Classifier on CelebA')
parser.add_argument('--data_dir', type=str, required=True, help='Path to CelebA images')
parser.add_argument('--label_path', type=str, required=True, help='Path to CelebA identity labels')
parser.add_argument('--output_dir', type=str, default='outputs/', help='Directory to save outputs')
parser.add_argument('--model_dir', type=str, default='models/', help='Directory of models')
parser.add_argument('--model_name', type=str, default='vit_linear_occluded', help='Model name for saving')
parser.add_argument('--max_classes', type=int, default=2000, help='Maximum number of identity classes to use')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--include_occlusions', action='store_true', help='Include occluded images')
parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

# ----------------- EarlyStopping -----------------
class EarlyStopping:
    def __init__(self, patience=3, delta=0, verbose=True, mode='max'):
        """
        Args:
            patience: no improvement for how many epochs
            delta: value of least improvement
            verbose: choose whether to print out the early stopping indicator
            mode: 'min' refers to monitoring the value that need to be minimised, e.g. loss;
                'max' refers to monitoring the value that need to be maximised, e.g. accuracy
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        self.best_epoch = 0
        
    def __call__(self, epoch, val_metric):
        score = val_metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (self.mode == 'min' and score >= self.best_score - self.delta) or \
             (self.mode == 'max' and score <= self.best_score + self.delta):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.best_score, self.best_epoch


# ----------------- Validation Function -----------------
def evaluate(model, loader, device, criterion):
    model.eval()
    correct, total, loss_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_total += loss.item()
    acc = correct / total
    return loss_total / total, acc


# ----------------- Main -----------------
def main():
    
    set_seed(args.seed)
    
    output_dir = os.path.join(args.output_dir, f"{args.model_name}")
    model_dir = args.model_dir
    log_dir = "runs/vit-occlusion"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # save config
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        config = vars(args)
        config['start_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
        json.dump(config, f, indent=4)
    
    # set model path
    model_path = os.path.join(model_dir, f"{args.model_name}_best.pth")
    final_model_path = os.path.join(model_dir, f"{args.model_name}_final.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # TensorBoard initialize
    writer = SummaryWriter(log_dir=log_dir)

    # data augmentation for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # data augmentation for evaluation
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 三通道
    ])

    # dataset loading
    try:
        train_dataset = CelebADataset(args.data_dir, args.label_path, transform=transform, 
                                   selected_ids=args.max_classes, 
                                   include_occlusions=args.include_occlusions)
        val_test_dataset = CelebADataset(args.data_dir, args.label_path, transform=eval_transform, 
                                     selected_ids=args.max_classes, 
                                     include_occlusions=args.include_occlusions)
        
        num_classes = len(set(train_dataset.labels))
        print(f"Number of classes: {num_classes}")
        
        # use seed to split the dataset
        generator = torch.Generator().manual_seed(args.seed)
        
        # 7:2:1
        train_size = int(0.7 * len(train_dataset))
        val_size = int(0.2 * len(train_dataset))
        test_size = len(train_dataset) - train_size - val_size
        
        train_set, val_set, test_set = random_split(
            train_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        print(f"Dataset splits - Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        
        train_loader = DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # ----------------- Model -----------------
    model = ViTLinearClassifier(num_classes=num_classes).to(device)
    
    # print model structure if you need
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()     # CE loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)   # AdamW optimizer
    
    # learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    start_epoch = 0
    best_acc = 0.0
    
    # resume/continue training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # check state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint.get('val_acc', 0.0)
                print(f"Resuming from epoch {start_epoch} with best val acc: {best_acc:.4f}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights only")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    early_stopper = EarlyStopping(patience=3, mode='max', verbose=True)
    
    # output sample training image in TensorBoard
    try:
        images, _ = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images[:16], normalize=True)
        writer.add_image('train_images', grid, 0)
    except Exception as e:
        print(f"Warning: Could not add image to tensorboard: {e}")

    # ----------------- Training Loop -----------------
    print(f"Starting training from epoch {start_epoch + 1} to {start_epoch + args.epochs}")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # progress bar using tqdm
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs}")
        
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # training accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss/(i+1):.4f}",
                'acc': f"{correct/total:.4f}"
            })
            
            # loss per 100 batch
            if i % 100 == 0:
                step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/train_step', loss.item(), step)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # validation
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        # update learning rate scheduler
        scheduler.step(val_acc)
        
        # record current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch)
        
        # update in TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # save best model
        best_score, best_epoch = early_stopper(epoch, val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes
            }, model_path)
            print(f"Best model saved to: {model_path} with accuracy: {val_acc:.4f}")
        
        
        # early stopping triggered
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1} with validation accuracy: {best_score:.4f}")
            break
    
    # save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'num_classes': num_classes
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # testing using BEST model
    print("\nEvaluating best model on test set...")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    
    # result on test set
    writer.add_scalar('Accuracy/test', test_acc, 0)
    writer.add_scalar('Loss/test', test_loss, 0)
    
    # save result
    results = {
        'best_val_acc': best_acc,
        'best_epoch': best_epoch + 1,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'total_epochs_trained': epoch + 1,
        'end_time': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    # 捕获可能的错误以提供更好的错误信息
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()
