import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Experiment_name')
    parser.add_argument('--model', default='unet', type=str, choices=['crnn', 'unet', 'unet2'])
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--max_epoch', default=200, type=int, help='Max epoch')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--threshold', default=0.55, type=float, help='Decision threshold')
    cfg = parser.parse_args()

    cfg.RR_MEAN = 641.282
    cfg.RR_STD = 121.321
    cfg.TIME_MEAN = 162468.
    cfg.TIME_STD = 235341.

    cfg.MAX_N_TICKS = 3661         # max n_measures

    cfg.MAX_LEN = 3840  # little padding for all
    cfg.MAX_OBS_TIME = 1900000
    cfg.TRAIN = 0.7
    cfg.VAL = 0.3
    return cfg
