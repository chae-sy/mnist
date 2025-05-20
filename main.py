import argparse
import torch
from datetime import datetime
from dataset import get_mnist_dataloaders
from model import MNISTNet
from train import train_epoch, validate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index (0-3) to use')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader = get_mnist_dataloaders(args.batch_size)

    model = MNISTNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer)
        val_loss, val_acc     = validate(model, device, val_loader)

        print(f'Epoch {epoch:02d}: '
              f'Train loss {train_loss:.4f}, acc {train_acc:.4f} | '
              f'Val loss {val_loss:.4f}, acc {val_acc:.4f}')

        # 검증 정확도 갱신 시마다 저장
        if val_acc > best_acc:
            best_acc = val_acc
            # 현재 날짜·시간 포맷: May20_1010
            now = datetime.now()
            ts = now.strftime("%B%d_%H%M")  # ex. "May20_1010"
            filename = f"Model_{ts}.pt"
            torch.save(model.state_dict(), filename)
            print(f"  → Saved new best model: {filename}")

if __name__ == '__main__':
    main()

