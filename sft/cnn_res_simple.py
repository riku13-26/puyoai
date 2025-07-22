# ------------------------------ #
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from pathlib import Path

# プロジェクトルート (= PUYO/) を取得
ROOT_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = ROOT_DIR / "data"

X_PATH = DATA_DIR / "X_dataset.npy"
Y_PATH = DATA_DIR / "Y_dataset.npy"

# ------------------------------ #
# 1.  ハイパーパラメータ設定
# ------------------------------ #
CONFIG = {
    # モデル
    'model_name':     'simple_residual_cnn',
    'n_blocks':        20,      # ResidualBlock 繰り返し回数
    'input_channels':  62,      
    'input_height':   14,      # 盤面縦
    'input_width':     6,      # 盤面横
    'num_classes':    17,      # 分類ならクラス数、回帰なら 1

    # 学習ハイパーパラメータ
    'batch_size':     512,
    'learning_rate':  1e-3,
    'num_epochs':      50,
    'scheduler_step_size': 100,
    'scheduler_gamma':     0.1,

    # データ
    'data_dir':       '/kaggle/input/puyo-data',
    'val_split':       0.2,

    # 乱数シード & デバイス
    'random_seed':     42,
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',

    # モデル保存
    'save_model':      True,
    'model_save_path': 'best_rescnn_model.pth',
}

# 固定シード
torch.manual_seed(CONFIG['random_seed'])
np.random.seed(CONFIG['random_seed'])

# ------------------------------ #
# 2.  モデル実装
# ------------------------------ #
class SimpleResidualBlock(nn.Module):
    """3×3 Conv ×2 + skip"""
    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResidualCNN(nn.Module):
    """
    図の全体構造:
        Conv(3×3,256) → BN → ReLU →
        ResidualBlock × n →
        Conv(1×1,1) →
        Flatten →
        FC(?,256) → ReLU → FC(out_dim)
    """
    def __init__(
        self,
        in_channels: int,
        n_blocks:    int,
        h:           int,
        w:           int,
        out_dim:     int
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.res_layers = nn.Sequential(
            *[SimpleResidualBlock(256) for _ in range(n_blocks)]
        )

        self.conv_head = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.spatial_do = nn.Dropout2d(0.2)
        
        self.flatten   = nn.Flatten()
        self.fc1       = nn.Linear(h * w, 256)
        self.fc_out    = nn.Linear(256, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_layers(x)
        
        x = self.conv_head(x)      # (N,1,H,W)
        # x = self.spatial_do(x)
        
        x = self.flatten(x)        # (N,H*W)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        return self.fc_out(x)
    
# ------------------------------ #
# 3.  データロード
# ------------------------------ #
def load_data(data_dir):
    X = np.load(X_PATH)  # (N,C,H,W)
    y = np.load(Y_PATH)  # (N,)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    # 正規化
    X[:, 12:14] /= 19.0
    return X, y


def create_loaders(X, y, batch_size, val_split):
    n = len(X)
    n_val = int(n * val_split)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ------------------------------ #
# 4.  学習・評価ループ
# ------------------------------ #
def train_epoch(model, loader, criterion, optim_, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in tqdm(loader, desc='Train', leave=False):
        data, target = data.to(device), target.to(device)
        optim_.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        optim_.step()

        total_loss += loss.item()
        pred = out.argmax(1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in tqdm(loader, desc='Val', leave=False):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss = criterion(out, target)

            total_loss += loss.item()
            pred = out.argmax(1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), correct / total

# ------------------------------ #
# 5.  メイン
# ------------------------------ #
def main():
    # W&B
    wandb.login()
    wandb.init(
        project='puyo-rescnn-training',
        config=CONFIG,
        name=f"rescnn_{CONFIG['n_blocks']}"
    )

    device = CONFIG['device']
    print(f'Using device: {device}')

    # データ
    X, y = load_data(CONFIG['data_dir'])
    train_ld, val_ld = create_loaders(X, y, CONFIG['batch_size'], CONFIG['val_split'])

    # モデル
    model = ResidualCNN(
        in_channels=CONFIG['input_channels'],
        n_blocks   =CONFIG['n_blocks'],
        h          =CONFIG['input_height'],
        w          =CONFIG['input_width'],
        out_dim    =CONFIG['num_classes']
    ).to(device)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Optimizer, Loss, Scheduler
    criterion = nn.CrossEntropyLoss()
    optim_ = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    sched  = optim.lr_scheduler.StepLR(
        optim_, step_size=CONFIG['scheduler_step_size'], gamma=CONFIG['scheduler_gamma']
    )

    # 学習ループ
    best_val_acc = 0
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    for epoch in range(CONFIG['num_epochs']):
        print(f'\nEpoch {epoch+1}/{CONFIG["num_epochs"]}')
        train_loss, train_acc = train_epoch(model, train_ld, criterion, optim_, device)
        val_loss, val_acc     = validate(model, val_ld, criterion, device)
        sched.step()

        # ログ
        wandb.log({
            'epoch': epoch+1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc':  train_acc,  'val_acc':  val_acc,
            'lr': sched.get_last_lr()[0]
        })

        train_loss_hist.append(train_loss); val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc);   val_acc_hist.append(val_acc)

        print(f'Train  loss:{train_loss:.4f} acc:{train_acc:.4f}')
        print(f'Val    loss:{val_loss:.4f} acc:{val_acc:.4f}')

        # モデル保存
        if val_acc > best_val_acc and CONFIG['save_model']:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim_.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, CONFIG['model_save_path'])
            print(f'  >> New best model saved: acc {val_acc:.4f}')

    # 学習曲線
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_loss_hist, label='train'); plt.plot(val_loss_hist, label='val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_acc_hist, label='train'); plt.plot(val_acc_hist, label='val')
    plt.title('Acc'); plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    wandb.log({'training_curves': wandb.Image('training_curves.png')})
    wandb.finish()
    print(f'\nBest validation accuracy: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()