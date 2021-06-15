import numpy as np
import torch
import torchvision.transforms as transforms


class Normalize(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, rr_mean_std, time_mean_std=None):
        assert isinstance(rr_mean_std, tuple)
        self.rr_mean, self.rr_std = rr_mean_std
        if time_mean_std is not None:
            assert isinstance(time_mean_std, tuple)
            self.t_mean, self.t_std = time_mean_std
        else:
            self.t_mean, self.t_std = None, None

    def __call__(self, sample):
        person = sample['person']
        person[1, :] -= self.rr_mean
        person[1, :] /= self.rr_std

        if self.t_mean is not None:
            person[0, :] -= self.t_mean
            person[0, :] /= self.t_std

        sample['person'] = person

        return sample


class PadZeros(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, max_len):
        assert isinstance(max_len, int)
        self.max_len = max_len

    def __call__(self, sample):
        person, labels = sample['person'], sample['labels']
        assert len(labels) <= self.max_len, 'Sequence length greater than allowed'

        person = np.pad(person, ((0, 0), (0, self.max_len - len(labels))), mode='constant', constant_values=0.)
        labels = np.pad(labels, (0, self.max_len - len(labels)), mode='constant', constant_values=0.)

        sample['person'] = person
        sample['labels'] = labels
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        person, labels = sample['person'], sample['labels']
        sample['person'] = torch.from_numpy(person)
        sample['labels'] = torch.from_numpy(labels)
        return sample


class ToSequenceTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        person, labels = sample['person'], sample['labels']
        sample['person'] = torch.from_numpy(person).permute(1, 0)  # DIM 1 IS TIME DIMENSION
        sample['labels'] = torch.from_numpy(labels)
        return sample


def get_base_transform(cfg):
    base_transform = transforms.Compose([
        Normalize((cfg.RR_MEAN, cfg.RR_STD), (0, cfg.RR_MEAN * cfg.MAX_N_TICKS / 2.)),
        PadZeros(cfg.MAX_LEN),
        ToTensor()
    ])
    return base_transform


def get_sequence_transform(cfg):
    base_transform = transforms.Compose([
        Normalize((cfg.RR_MEAN, cfg.RR_STD), (0, cfg.RR_MEAN * cfg.MAX_N_TICKS / 2.)),
        ToSequenceTensor()
    ])
    return base_transform
