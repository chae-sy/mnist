import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

def train_epoch(model, device, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in tqdm(loader, desc='Train', leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def validate(model, device, loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

