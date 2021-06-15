from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import torch
import random

from data.dataset import BaseDataset
from data.transforms import get_base_transform
from options import get_config
from torch.utils.data import DataLoader
from models.CRNN import CRNN
from models.loss import BCELoss
import os
from eval import evaluate_metrics, AverageMeter
from dynamic_ecg import FigPlotter


# FIX RANDOM SEED
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

TRAIN_STEP = 0
PLOTTER = FigPlotter()


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
    print(f'Training epoch {epoch}')
    global TRAIN_STEP
    model.train()
    avg_loss = AverageMeter()

    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # BATCH CROP LEN
        person, labels, seq_lens = sample['person'], sample['labels'], sample['end_pos']
        max_seq_len = max(seq_lens)
        if max_seq_len % 4:
            max_seq_len += (4 - max_seq_len % 4)
        person = person[..., : max_seq_len]
        labels = labels[..., : max_seq_len]
        # FORWARD
        output = model(person.float().to(device))
        # LOSS
        loss = criterion(output, labels.to(device))
        avg_loss.update(loss.item())
        # BAKCWARD
        optimizer.zero_grad()
        loss.backward()
        # STEP
        optimizer.step()
        TRAIN_STEP += 1

        writer.add_scalar('train/loss', avg_loss.avg, global_step=TRAIN_STEP)
        writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=TRAIN_STEP)
    scheduler.step()
    print(f'Average Train Loss: {avg_loss.avg}')


def validate(model, val_loader, criterion, epoch, device, writer, threshold):
    print(f'Validate epoch {epoch}')
    with torch.no_grad():
        model.eval()
        avg_loss = AverageMeter()
        all_outputs = []
        all_labels = []

        for i, sample in tqdm(enumerate(val_loader), total=len(val_loader)):
            # BATCH CROP LEN
            person, labels, seq_lens = sample['person'], sample['labels'], sample['end_pos']
            max_seq_len = max(seq_lens)
            if max_seq_len % 4:
                max_seq_len += (4 - max_seq_len % 4)
            person = person[..., : max_seq_len]
            labels = labels[..., : max_seq_len]

            # FORWARD
            output = model(person.float().to(device))
            # LOSS
            loss = criterion(output, labels.to(device))
            avg_loss.update(loss.item())
            # TODO: NEED TO HANDLE LENGTH TOO!!!
            # OUTPUT IS [BATCH x TIME] & [LABELS BATCH x TIME]
            output = (output.sigmoid() > threshold).int().cpu().numpy()
            labels = labels.int().cpu().numpy()
            PLOTTER.plot_ecg(person[0, 0, :],
                             person[0, 1, :],
                             labels[0],
                             output[0],
                             sample['person_id'][0])

            all_outputs += list(output.flatten())
            all_labels += list(labels.flatten())

        print(f"final shapes: labels = {len(all_outputs)}, output = {len(all_labels)}")
        score = evaluate_metrics(all_outputs, all_labels)
        print(f'Average Val Loss: {avg_loss.avg}')
        print(f'Metrics score: {score}')

        writer.add_scalar('val/loss', avg_loss.avg, global_step=epoch)
        writer.add_scalar('val/score', score, global_step=epoch)
        writer.add_figure('val/rr', PLOTTER.get_figures(), global_step=epoch)
        PLOTTER.refresh()

        return avg_loss.avg, score


def main(cfg):
    train_loader, val_loader = init_dataset(cfg)
    model = get_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.)  # CONSTANT LEARNING RATE
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
            print(f'Best val score {best_val_score} achieved on epoch {best_epoch}.')
            print('*****')


if __name__ == '__main__':
    cfg = get_config()
    cfg.device = torch.device('cuda' if torch.cuda.is_available() and cfg.cuda else 'cpu')
    cfg.writer = SummaryWriter(os.path.join('experiments', cfg.experiment_name, 'log'))
    main(cfg)
