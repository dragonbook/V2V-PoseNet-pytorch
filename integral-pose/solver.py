import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from progressbar import progress_bar


def train_epoch(model, criterion, optimizer, train_loader, device=torch.device('cuda'), dtype=torch.float, collector=None):
    model.train()
    train_loss = 0

    for batch_idx, batch_data in enumerate(train_loader):
        input, target, extra = batch_data['input'], batch_data['target'], batch_data['extra']

        input = input.to(device, dtype)

        if isinstance(target, torch.Tensor):
            target = target.to(device, dtype)
        elif isinstance(target, dict):
            for k in target:
                if isinstance(target[k], torch.Tensor):
                    target[k] = target[k].to(device, dtype)

        #print('solver target[heatmap]: ', target['heatmap'].dtype)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if collector is None:
            train_loss += loss.item()
            progress_bar(batch_idx, len(train_loader), 'Loss: {0:.4e}'.format(train_loss/(batch_idx+1)))
            #print('loss: {0: .4e}'.format(train_loss/(batch_idx+1)))
        else:
            model.eval()
            with torch.no_grad():
                extra['batch_idx'], extra['loader_len'], extra['batch_avg_loss'] = batch_idx, len(train_loader), loss.item()
                collector({'model': model, 'input': input, 'target': target, 'output': output, 'extra': extra})
            
            # Keep train mode
            model.train()


def val_epoch(model, criterion, val_loader, device=torch.device('cuda'), dtype=torch.float, collector=None):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            input, target, extra = batch_data['input'], batch_data['target'], batch_data['extra']

            input = input.to(device, dtype)

            if isinstance(target, torch.Tensor):
                target = target.to(device, dtype)
            elif isinstance(target, dict):
                for k in target:
                    if isinstance(target[k], torch.Tensor):
                        target[k] = target[k].to(device, dtype)

            output = model(input)
            loss = criterion(output, target)

            if collector is None:
                val_loss += loss.item()
                progress_bar(batch_idx, len(val_loader), 'Loss: {0:.4e}'.format(val_loss/(batch_idx+1)))
                #print('loss: {0: .4e}'.format(val_loss/(batch_idx+1)))
            else:
                extra['batch_idx'], extra['loader_len'], extra['batch_avg_loss'] = batch_idx, len(val_loader), loss.item()
                collector({'model': model, 'input': input, 'target': target, 'output': output, 'extra': extra})

                # Keep eval mode
                model.eval()


def test_epoch(model, test_loader, collector, device=torch.device('cuda'), dtype=torch.float):
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            input, target, extra = batch_data['input'], batch_data['target'], batch_data['extra']
            output = model(input.to(device, dtype))

            extra['batch_idx'], extra['loader_len'] = batch_idx, len(test_loader)
            collector({'model': model, 'input': input, 'target': target, 'output': output, 'extra': extra})

            # Keep eval mode
            model.eval()


class Solver():
    def __init__(self, train_set, model, criterion, optimizer, device=torch.device('cuda'), dtype=torch.float, **kwargs):
        self.train_set = train_set
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.dtype = dtype

        self.batch_size = kwargs.pop('batch_size', 1)
        self.num_epochs = kwargs.pop('num_epochs', 1)
        self.num_workers = kwargs.pop('num_workers', 6)

        self.val_set = kwargs.pop('val_set', None)

        # Result collectors
        self.train_collector = kwargs.pop('train_collector', None)
        self.val_collector = kwargs.pop('val_collector', None)

        # Save check point and resume
        self.checkpoint_config = kwargs.pop('checkpoint_config', None)
        if self.checkpoint_config is not None:
            self.save_checkpoint = self.checkpoint_config['save_checkpoint']
            self.checkpoint_dir = self.checkpoint_config['checkpoint_dir']
            self.checkpoint_per_epochs = self.checkpoint_config['checkpoint_per_epochs']

            self.resume_training = self.checkpoint_config['resume_training']
            self.resume_after_epoch = self.checkpoint_config['resume_after_epoch']
        else:
            self.save_checkpoint = False
            self.resume_training = False

        self._init()

    def _init(self):
        self.train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

        if self.val_set is not None:
            self.val_loader = DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

        self.start_epoch = 0

        if self.resume_training:
            self._load_checkpoint(self.resume_after_epoch)

    def _train_epoch(self):
        train_epoch(self.model, self.criterion, self.optimizer, self.train_loader,
                    self.device, self.dtype,
                    self.train_collector) 

    def _val_epoch(self):
        val_epoch(self.model, self.criterion, self.val_loader,
                  self.device, self.dtype,
                  self.val_collector)

    def _save_checkpoint(self, epoch):
        if not os.path.exists(self.checkpoint_dir): os.mkdir(self.checkpoint_dir)
        checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch'+str(epoch)+'.pth')

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(checkpoint, checkpoint_file)   

    def _load_checkpoint(self, epoch):
        checkpoint_file = os.path.join(self.checkpoint_dir, 'epoch'+str(epoch)+'.pth')

        print('==> Resuming from checkpoint after epoch {} ..'.format(epoch))
        assert os.path.isdir(self.checkpoint_dir), 'Error: no checkpoint directory found!'
        assert os.path.isfile(checkpoint_file), 'Error: no checkpoint file of epoch {}'.format(epoch)

        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'epoch'+str(epoch)+'.pth'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            print('Epoch {}: '.format(epoch))
            self._train_epoch()

            if self.val_set is not None:
                self._val_epoch()

            if self.save_checkpoint and epoch % self.checkpoint_per_epochs == 0:
                self._save_checkpoint(epoch)
