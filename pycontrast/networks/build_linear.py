import torch.nn as nn


def build_linear(opt):
    n_class = opt.n_class
    arch = opt.arch
    if arch.endswith('x4'):
        n_feat = 2048 * 4
    elif arch.endswith('x2'):
        n_feat = 2048 * 2
    elif arch == 'resnet18':
        n_feat = 512
    else:
        n_feat = 2048
    import os
    if 'SKIP_POOL' in os.environ:
        n_feat = n_feat * 7 * 7

    classifier = nn.Linear(n_feat, n_class)
    return classifier
