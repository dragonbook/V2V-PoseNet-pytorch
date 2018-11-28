import torch
import torch.optim as optim
from progressbar import progress_bar


def train_epoch(model, criterion, optimizer, train_loader, device=torch.device('cuda'), dtype=torch.float):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, dtype)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.3f}'.format(train_loss/(batch_idx+1)))


def test_epoch(model, criterion, test_loader, device=torch.device('cuda'), dtype=torch.float):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            progress_bar(batch_idx, len(test_loader), 'Loss: {0:.3f}'.format(test_loss/(batch_idx+1)))
