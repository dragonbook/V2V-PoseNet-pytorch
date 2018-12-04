import torch
import torch.optim as optim
from lib.progressbar import progress_bar


def train_epoch(model, criterion, optimizer, train_loader, device=torch.device('cuda'), dtype=torch.float):
    model.train()
    train_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, dtype), targets.to(device, dtype)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
        #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))


def val_epoch(model, criterion, val_loader, device=torch.device('cuda'), dtype=torch.float):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device, dtype), targets.to(device, dtype)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: {0:.4e}'.format(val_loss/(batch_idx+1)))
            #print('loss: {0: .4e}'.format(val_loss/(batch_idx+1)))


def test_epoch(model, test_loader, result_collector, device=torch.device('cuda'), dtype=torch.float):
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, extra) in enumerate(test_loader):
            outputs = model(inputs.to(device, dtype))
            result_collector((inputs, outputs, extra))
