import torch


# ======================
# Evaluate
# ======================
@torch.no_grad()
def evaluate(model,
             loader,
             criterion,
             device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)

        labels = labels.to(device)

        cls_logits, dist_logits = model(images)

        loss = criterion(cls_logits, labels)

        total_loss += loss.item()

        _, predicted = cls_logits.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total

    avg_loss = total_loss / len(loader)

    return avg_loss, acc


# ======================
# Save Model
# ======================
def save_checkpoint(model,
                    path):

    torch.save(model.state_dict(), path)

    print(f'Saved model to {path}')