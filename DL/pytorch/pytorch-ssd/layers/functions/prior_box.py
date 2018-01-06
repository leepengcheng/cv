# coding:utf-8
from itertools import product as product
from math import sqrt as sqrt

import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    f_k = self.image_size / self.steps[k]  # 300/8=37.5
                    # unit center x,y
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # 正方形box,宽度:sk
                    s_k = self.min_sizes[k] / self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # 正方形box,宽度: sqrt(s_k * s_(k+1))
                    s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # 生成其他宽高比的box,每个宽aspect_ratio生成2个长方形
                    # 宽高分别为sk*sqrt(aspect_ratio),sk/sqrt(aspect_ratio)
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                        mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                # 步长:featuremap每格对应的输入图片的大小
                step_x = step_y = self.image_size / k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w + 0.5) * step_x) #映射回原图中的位置x
                    c_y = ((h + 0.5) * step_y) #映射回原图中的位置y
                    c_w = c_h = self.min_sizes[i] / 2 #宽度和高度为min_size的一半
                    s_k = self.image_size  # 300
                    # Box 1:aspect_ratio: 1,size: min_size
                    # 正方形窗口1,1box的左上角和右下角位置的归一化
                    mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k,
                             (c_x + c_w) / s_k, (c_y + c_h) / s_k]
                    if self.max_sizes[i] > 0:
                        # Box 2:aspect_ratio: 1,size: sqrt(min_size * max_size)/2
                        # 正方形窗口2,1box的左上角和右下角位置的归一化
                        c_w = c_h = sqrt(self.min_sizes[i] * self.max_sizes[i]) / 2
                        mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k,
                                 (c_x + c_w) / s_k, (c_y + c_h) / s_k]
                    # 其余宽高比的prior box
                    for ar in self.aspect_ratios[i]:
                        #ar 不为1时(宽高比为1的上面已经算了2个)
                        if not (abs(ar - 1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar) / 2
                            c_h = self.min_sizes[i] / sqrt(ar) / 2
                            mean += [(c_x - c_w) / s_k, (c_y - c_h) / s_k,
                                     (c_x + c_w) / s_k, (c_y + c_h) / s_k]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
