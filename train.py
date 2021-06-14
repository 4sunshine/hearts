from options import get_config
import numpy as np
import torch
# FIX RANDOM SEED
np.random.seed(42)

from data.dataset import BaseDataset
from data.transforms import get_base_transform
from torch.utils.data import DataLoader
from models.CRNN import CRNN
from models.loss import BCELoss
from tensorboardX import SummaryWriter
import os
from eval import evaluate_metrics

train_step = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_model(cfg):
    if cfg.model == 'crnn':
        model = CRNN(num_class=1)
    else:
        raise NotImplementedError(f'Model {cfg.model} currently not implemented')
    if cfg.resume:
        model.load_state_dict(torch.load(cfg.resume_path))
    return model.to(cfg.device)


def init_dataset(cfg):
    train_transform = get_base_transform(cfg)
    val_transform = get_base_transform(cfg)
    train_set = BaseDataset(is_train=True, transform=train_transform, cfg=cfg)
    val_set = BaseDataset(is_train=False, transform=val_transform, cfg=cfg)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train(model, train_loader, criterion, scheduler, optimizer, epoch, device, writer):
    global train_step
    model.train()
    scheduler.step()
    print(f'Training epoch {epoch}')
    avg_loss = AverageMeter()

    for i, sample in enumerate(train_loader):
        person, labels = sample['person'], sample['labels']
        output = model(person.to(device))
        loss = criterion(output, labels.to(device))
        loss.backward()
        optimizer.step()
        train_step += 1

        avg_loss.update(loss.item())
        writer.add_scalars({
            'loss': loss,
            'learning_rate': scheduler.get_lr(),
        }, tag='train', n_iter=train_step)

    print(f'Average Train Loss: {avg_loss.avg}')


def validate(model, val_loader, criterion, epoch, device, writer, threshold):
    with torch.no_grad():
        model.eval()
        print(f'Validate epoch {epoch}')
        avg_loss = AverageMeter()
        all_outputs = []
        all_labels = []

        for i, sample in enumerate(val_loader):
            person, labels = sample['person'], sample['labels']
            output = model(person.to(device))
            loss = criterion(output, labels.copy().to(device))
            avg_loss.update(loss.item())
            output = (output.sigmoid() > threshold).int().cpu().numpy()
            # NEED TO HANDLE LENGTH TOO!!!
            # OUTPUT IS [BATCH x TIME] & [LABELS BATCH x TIME]
            all_outputs.append(output)
            all_labels.append(labels)

        final_output = np.concatenate(all_outputs, axis=0).flatten()
        final_labels = np.concatenate(all_labels, axis=0).flatten()
        score = evaluate_metrics(final_output, final_labels)

        writer.add_scalars({
            'loss': avg_loss.avg,
            'score': score,
        }, tag='val', n_iter=train_step)

        print(f'Average Val Loss: {avg_loss.avg}')
        print(f'Metrics score: {score}')

        return avg_loss.avg, score


def main(cfg):
    train_loader, val_loader = init_dataset(cfg)
    model = get_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.) # CONSTANT LEARNING RATE
    criterion = BCELoss()
    save_dir = os.path.join('experiments', cfg.experiment_name)
    best_val_score = 0.
    best_epoch = 0

    for epoch in range(cfg.max_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch, cfg.device, cfg.writer)
        val_loss, val_score = validate(model, val_loader, criterion, epoch, cfg.device, cfg.writer, cfg.threshold)
        if val_score >= best_val_score:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_score.pth'))
            print(f'Best val score {val_score} achieved on epoch {epoch}.')
            best_val_score = val_score
            best_epoch = epoch
            print('*****')
        else:
            print(f'Epoch {epoch} passed. Val score is {val_score}.')
            print(f'Best val score {val_score} achieved on epoch {best_epoch}.')
            print('*****')


if __name__ == '__main__':
    cfg = get_config()
    cfg.device = torch.device('cuda' if torch.cuda.is_available() and cfg.cuda else 'cpu')
    cfg.writer = SummaryWriter(os.path.join('experiments', cfg.experiment_name, 'log'))
    main(cfg)
