from torch.nn.modules import Module
import torch
from math import ceil

class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    '''
    @p prob_list [torch.size([6, 1024]), ...]
    @p target_list [torch.size([5]), ...]
    @p pre_density torch.size([1, 1, 32, 32])
    '''
    def forward(self, prob_list, target_list, pre_density):
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                # prob的长度表示注释的点数 use_background=True时，prob比target的维度多一维
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            res = torch.abs(target - pre_count)
            # 向上取整
            num = ceil(0.9 * (len(res) - 1))
            loss += torch.sum(torch.topk(res[:-1], num, largest=False)[0])
            loss += res[-1]
        loss = loss / len(prob_list)
        return loss



