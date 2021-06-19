from train import get_model
from data.transforms import get_base_sequence_transform
from data.dataset import TestDataset
from torch.utils.data import DataLoader
from options import get_config


if __name__ == '__main__':
    cfg = get_config()
    model = get_model(cfg)
    val_transform = get_base_sequence_transform(cfg)
    test_set = TestDataset(transform=val_transform, cfg=cfg)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)



