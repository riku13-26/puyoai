import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import wandb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

wandb.login()

# プロジェクトルート (= PUYO/) を取得
ROOT_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT_DIR / "data"

X_PATH = DATA_DIR / "X_dataset.npy"
Y_PATH = DATA_DIR / "Y_dataset.npy"

X_train = np.load(X_PATH)  # (N, C, W, H)
y_train = np.load(Y_PATH)  # (N,)
NUM_CLASSES = int(np.max(y_train) + 1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("num_class:", NUM_CLASSES)
print(X_train[0].shape, y_train[0])
BATCH_SIZE = 256

CONFIG = {
    # Model architecture settings
    'model_name': 'resnet',                           # モデル名（ログ用）
    'num_blocks': [3, 4, 36, 3],                       # 各レイヤーのブロック数 [layer1, layer2, layer3, layer4]
    'bottleneck': True,
    
    # Training hyperparameters
    'batch_size': BATCH_SIZE,                                # バッチサイズ（メモリに応じて調整）
    'learning_rate': 0.001,                           # 学習率（通常0.001-0.1）
    'num_epochs': 50,                                 # 学習エポック数
    
    # Learning rate scheduler settings
    'scheduler_step_size': 100,                        # 学習率減衰のステップ間隔（エポック）
    'scheduler_gamma': 0.1,                           # 学習率減衰率（0.1-0.9）
    
    # Data settings
    'input_channels': 62,                             # 入力チャンネル数（14チャンネル）
    'input_width': 14,                                # 入力画像幅
    'input_height': 6,                                # 入力画像高さ
    'num_classes': 17,                                # 分類クラス数（行動数）
    
    # System settings
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 使用デバイス（GPU/CPU）
    'val_split': 0.2,                                 # 検証データの割合（0.1-0.3）
    'random_seed': 42,                                # 再現性のための乱数シード
    
    # Model saving settings
    'save_model': True,                               # モデル保存フラグ
    'model_save_path': 'best_resnet_model.pth'        # 最良モデルの保存パス
}

# Example configurations for different ResNet variants:
# ResNet-18: num_blocks = [2, 2, 2, 2]
# ResNet-34: num_blocks = [3, 4, 6, 3]
# ResNet-50: num_blocks = [3, 4, 6, 3] with Bottleneck blocks
# ResNet-101: num_blocks = [3, 4, 23, 3] with Bottleneck blocks
# ResNet-152: num_blocks = [3, 4, 36, 3] with Bottleneck blocks
# ResNet-200: num_blocks = [3, 24, 36, 3] with Bottleneck blocks

# Set random seed
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# -------------------------------------------------------------
# 1. BasicBlock  (ResNet-18 / 34 用)
# -------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------------------------------------------
# 2. Bottleneck (ResNet-50 / 101 / 152 用)  ★ new
# -------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4          # 出力チャネルは planes * 4
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1×1 でチャネル圧縮
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        # 3×3 で処理（空間解像度を落とす場合はここで stride）
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # 1×1 でチャネル拡張
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        
        # ショートカット（形状合わせ）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------------------------------------------
# 3. 共通の ResNet 本体
# -------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=17, input_channels=14):
        super().__init__()
        self.in_planes = 64
        
        # stem
        self.conv1 = nn.Conv2d(input_channels, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512 * block.expansion, num_classes)

        self.conv_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.flatten   = nn.Flatten()
        self.fc1       = nn.Linear(14 * 6, 256)
        self.fc_out    = nn.Linear(256, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.avgpool(out)
        # out = torch.flatten(out, 1)
        # out = self.dropout(out)
        # return self.fc(out)

        out = self.conv_head(out)
        out = self.flatten(out)        # (N,H*W)
        out = F.relu(self.fc1(out))
        out = self.dropout(out) 
        return self.fc_out(out)

# -------------------------------------------------------------
# 4. 便利なラッパー関数
# -------------------------------------------------------------
def create_resnet(num_blocks, num_classes=17, input_channels=14, bottleneck=True):
    """
    Args:
        num_blocks   : 各ステージのブロック数 [n1, n2, n3, n4]
        num_classes  : クラス数
        input_channels: 入力チャネル数
        bottleneck   : True → Bottleneck, False → BasicBlock
    """
    block_type = Bottleneck if bottleneck else BasicBlock
    return ResNet(block_type, num_blocks, num_classes, input_channels)


def load_data():
    """Load and preprocess the Puyo game data"""
    # DATA_DIR = "/kaggle/input/puyo-data"
    
    # # Load data
    # X_train = np.load(os.path.join(DATA_DIR, "X_dataset.npy"))  # (N, C, W, H) = (N, 14, 14, 6)
    # y_train = np.load(os.path.join(DATA_DIR, "Y_dataset.npy"))  # (N,)
    
    # Convert to PyTorch tensors (already in correct format)
    # 正規化
    X_train[:, 12:14] /= 19.0
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    
    print(f"Data shape (N, C, W, H): {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    print(f"Number of classes: {torch.max(y_train) + 1}")
    
    return X_train, y_train

def create_data_loaders(X, y, batch_size, val_split=0.2):
    """Create training and validation data loaders"""
    
    # Split data
    n_samples = len(X)
    n_val = int(n_samples * val_split)
    
    # Random split
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    return total_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validating"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total

def main():
    # Initialize wandb
    wandb.init(
        project="puyo-resnet-training",
        config=CONFIG,
        name=f"resnet_blocks_{CONFIG['num_blocks']}"
    )
    
    print("Starting Puyo Game ResNet Training...")
    print(f"Using device: {CONFIG['device']}")
    print(f"Model configuration: {CONFIG['num_blocks']} blocks per layer")
    
    # Load data
    X, y = load_data()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X, y, CONFIG['batch_size'], CONFIG['val_split']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_resnet(
        num_blocks=CONFIG['num_blocks'],
        num_classes=CONFIG['num_classes'],
        input_channels=CONFIG['input_channels'],
        bottleneck=CONFIG['bottleneck']
    ).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=CONFIG['scheduler_step_size'],
        gamma=CONFIG['scheduler_gamma']
    )
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['num_epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, CONFIG['device'])
        
        # Update scheduler
        scheduler.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if CONFIG['save_model']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'config': CONFIG
                }, CONFIG['model_save_path'])
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    wandb.log({"training_curves": wandb.Image('training_curves.png')})
    
    # Final evaluation
    print("\nFinal Model Evaluation:")
    final_val_loss, final_val_acc = validate(model, val_loader, criterion, CONFIG['device'])
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    wandb.log({
        'final_val_accuracy': final_val_acc,
        'best_val_accuracy': best_val_acc
    })
    
    wandb.finish()

if __name__ == "__main__":
    main()

