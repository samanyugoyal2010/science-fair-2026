import math

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(model, dataset, batch_size: int, device: torch.device, max_batches: int = 50):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    val_loss = float(sum(losses) / max(1, len(losses)))
    val_ppl = float(math.exp(min(20, val_loss)))
    return val_loss, val_ppl
