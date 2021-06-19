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
        person, labels, mask = sample['person'], sample['labels'], sample['mask']
        assert len(labels) <= self.max_len, 'Sequence length greater than allowed'

        person = np.pad(person, ((0, 0), (0, self.max_len - len(labels))), mode='constant', constant_values=0.)
        labels = np.pad(labels, (0, self.max_len - len(labels)), mode='constant', constant_values=0.)
        mask = np.pad(mask, (0, self.max_len - len(labels)), mode='constant', constant_values=0.)

        sample['person'] = person
        sample['labels'] = labels
        sample['mask'] = mask
        return sample


class SigmaCrop(object):
    def __call__(self, sample):
        person = sample['person']

        if len(person > 200): # mediana
            mean, std = person[1].mean(), person[1].std()
            person[1][person[1] > (mean + 4 * std)] = mean + 4 * std
            person[1][person[1] < (mean - 4 * std)] = mean - 4 * std
            sample['person'] = person
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        person, labels, mask = sample['person'], sample['labels'], sample['mask']
        sample['person'] = torch.from_numpy(person)
        sample['labels'] = torch.from_numpy(labels)
        sample['mask'] = torch.from_numpy(mask)
        return sample


class RandomCrop(object):
    """Convert ndarrays in sample to Tensors."""
    # !!! APPLY IT ONLY AFTER ALL OTHER AUGMENTATIONS
    def __init__(self, probability):
        assert 0 <= probability <= 1
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            person, labels, mask = sample['person'], sample['labels'], sample['mask']
            start_inds = sample['anomalies_starts']
            end_inds = sample['anomalies_ends']
            seq_len = sample['end_pos']
            n_anomalies = len(start_inds)
            first_anomaly_to_hold = np.random.randint(n_anomalies)
            last_anomaly_to_hold = np.random.randint(first_anomaly_to_hold, n_anomalies)
            if first_anomaly_to_hold == 0:
                if start_inds[0] > 0:
                    crop_begin = np.random.randint(start_inds[0])
                else:
                    crop_begin = 0
            else:
                crop_begin = np.random.randint(end_inds[first_anomaly_to_hold - 1], start_inds[first_anomaly_to_hold])
            if last_anomaly_to_hold == (n_anomalies - 1):
                if end_inds[-1] == seq_len:
                    crop_end = seq_len
                else:
                    crop_end = np.random.randint(end_inds[-1], seq_len)
            else:
                crop_end = np.random.randint(end_inds[last_anomaly_to_hold],
                                             start_inds[last_anomaly_to_hold + 1])
            crop_length = crop_end - crop_begin
            if crop_length < 7:
                return sample
            sample['person'] = person[:, crop_begin: crop_end]
            sample['labels'] = labels[crop_begin: crop_end]
            sample['mask'] = mask[crop_begin: crop_end]
            start_inds = start_inds[first_anomaly_to_hold: last_anomaly_to_hold + 1]
            end_inds = end_inds[first_anomaly_to_hold: last_anomaly_to_hold + 1]
            sample['anomalies_starts'] = start_inds - crop_begin
            sample['anomalies_ends'] = end_inds - crop_begin
            sample['end_pos'] = crop_length
            return sample
        else:
            return sample


class ToSequenceTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        person, labels, mask = sample['person'], sample['labels'], sample['mask']
        sample['person'] = torch.from_numpy(person).permute(1, 0)  # DIM 1 IS TIME DIMENSION
        sample['labels'] = torch.from_numpy(labels)
        sample['mask'] = torch.from_numpy(mask)
        return sample


class ToTestSequenceTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        person = sample['person']
        sample['person'] = torch.from_numpy(person).permute(1, 0)  # DIM 1 IS TIME DIMENSION
        return sample


class RandomNoise(object):
    """Convert ndarrays in sample to Tensors."""
    # !!! APPLY IT ONLY AFTER ALL OTHER AUGMENTATIONS
    def __init__(self, probability, coefficient):
        assert 0 <= probability <= 1
        self.probability = probability
        self.coeff = coefficient

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            person, labels, mask = sample['person'], sample['labels'], sample['mask']
            start_inds = sample['anomalies_starts']
            end_inds = sample['anomalies_ends']
            amplitudes = []
            for st, en in zip(start_inds, end_inds):
                current_anomaly = person[1, st: en]
                amplitudes.append(np.max(current_anomaly) - np.min(current_anomaly))

            min_amplitude = min(amplitudes)
            noise_amplitude = self.coeff * min_amplitude
            noise = noise_amplitude * np.random.normal(0, 1, len(labels))
            noise = np.clip(noise, - 2 * noise_amplitude, 2 * noise_amplitude)
            person[1, :] += noise
            time_warp = person[1, 1:]
            measure_time = np.cumsum(time_warp)
            person[0, 1:] = measure_time
            sample['person'] = person
            return sample
        else:
            return sample


class RandomReflect(object):
    """Convert ndarrays in sample to Tensors."""
    # !!! APPLY IT ONLY AFTER ALL OTHER AUGMENTATIONS
    def __init__(self, probability):
        assert 0 <= probability <= 1
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            person, labels, mask = sample['person'], sample['labels'], sample['mask']
            start_inds = sample['anomalies_starts']
            end_inds = sample['anomalies_ends']
            seq_len = sample['end_pos']
            new_start_inds = seq_len - np.flip(end_inds.copy())
            new_end_inds = seq_len - np.flip(start_inds.copy())
            labels = np.flip(labels)
            person[1, :] = np.flip(person[1].copy())
            person[0, :] = person[0, -1] - np.flip(person[0].copy())
            sample['person'] = person.copy()
            sample['labels'] = labels.copy()
            sample['anomalies_starts'] = new_start_inds.copy()
            sample['anomalies_ends'] = new_end_inds.copy()
            return sample
        else:
            return sample


class RandomSlope(object):
    """Convert ndarrays in sample to Tensors."""
    # !!! APPLY IT ONLY AFTER ALL OTHER AUGMENTATIONS
    def __init__(self, probability, amplitude):
        assert 0 <= probability <= 1
        self.probability = probability
        self.amplitude = amplitude

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            person, labels, mask = sample['person'], sample['labels'], sample['mask']
            max_val, min_val = np.max(person[1, :]), np.min(person[1, :])
            dy = max_val - min_val
            sign = -1 if np.random.rand() < 0.5 else 1
            b = -self.amplitude * 0.5 * dy
            k = -2 * b / sample['end_pos']
            linsp = np.arange(sample['end_pos'])
            add_slope = k * linsp + b
            person[1, :] += sign * add_slope
            time_warp = person[1, 1:]
            measure_time = np.cumsum(time_warp)
            person[0, 1:] = measure_time
            sample['person'] = person
            return sample
        else:
            return sample


class RandomFlip(object):
    """Convert ndarrays in sample to Tensors."""
    # !!! APPLY IT ONLY AFTER ALL OTHER AUGMENTATIONS
    def __init__(self, probability):
        assert 0 <= probability <= 1
        self.probability = probability

    def __call__(self, sample):
        if np.random.rand() < self.probability:
            person, labels, mask = sample['person'], sample['labels'], sample['mask']
            max_val, min_val = np.max(person[1, :]), np.min(person[1, :])
            person[1, :] = min_val + (max_val - person[1, :])
            time_warp = person[1, 1:]
            measure_time = np.cumsum(time_warp)
            person[0, 1:] = measure_time
            sample['person'] = person
            return sample
        else:
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
        #SigmaCrop(),
        # RandomSlope(probability=0.3, amplitude=0.2),
        RandomNoise(probability=0.3, coefficient=0.2),
        RandomCrop(probability=0.3),
        #RandomReflect(probability=0.3),
        ToSequenceTensor(),
    ])
    return base_transform


def get_base_sequence_transform(cfg):
    base_transform = transforms.Compose([
        Normalize((cfg.RR_MEAN, cfg.RR_STD), (0, cfg.RR_MEAN * cfg.MAX_N_TICKS / 2.)),
        #SigmaCrop(),
        ToSequenceTensor()
    ])
    return base_transform


def get_test_sequence_transform(cfg):
    base_transform = transforms.Compose([
        Normalize((cfg.RR_MEAN, cfg.RR_STD), (0, cfg.RR_MEAN * cfg.MAX_N_TICKS / 2.)),
        #SigmaCrop(),
        ToTestSequenceTensor()
    ])
    return base_transform
