import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    计算连续 Lovász 扩展的梯度 (Jaccard 误差的交并比惩罚)
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # 差分计算
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probs, labels, classes='present'):
    """
    针对展平(Flatten)后的预测概率和标签计算 Lovász-Softmax 损失
    :param probs: [N, C] 经过 Softmax 激活的预测概率
    :param labels: [N] 真实标签
    :param classes: 'all' 表示计算所有类别，'present' 表示只计算当前 Batch 中出现的类别
    """
    if probs.numel() == 0:
        # 当 mask 为空时，返回 0 梯度
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # 前景掩码
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]

        # 计算该类别的逐点误差
        errors = (Variable(fg) - class_pred).abs()
        # 对误差进行降序排列
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        # 乘以 Lovász 梯度
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))

    if len(losses) == 0:
        return probs.sum() * 0.
    return sum(losses) / len(losses)


class BoundaryAwareJointLoss(nn.Module):
    def __init__(self, lambda_weight=0.5, sigma=2.0, gamma=2.0, tau=5.0):
        """
        边界感知联合损失函数 (Boundary-Aware Joint Loss)
        :param lambda_weight: 平衡系数 (论文中设为 0.5)
        :param sigma: 指数衰减速率，匹配未截断的 distance_field 分布
                      过渡区 D~2mm，非边界区 D>>20mm
                      exp(-2/2)=0.37 (边界), exp(-20/2)=4.5e-5 (远离)
        :param gamma: 边界惩罚增益系数
        :param tau: 边界带掩码阈值，仅 D<5mm 真正靠近边界的点进入 L_BLS
        """
        super(BoundaryAwareJointLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau
        weights = torch.tensor([1.0, 1.2, 3.0, 8.0, 1.2], dtype=torch.float32)
        self.register_buffer('class_weights', weights)

    def forward(self, logits, targets, distance_field):
        """
        :param logits: 网络预测的原始得分 [Batch, N, C_classes] (即咱们刚刚跑出来的 [2, 10000, 5])
        :param targets: 真实标签 [Batch, N] (即 [2, 10000])
        :param distance_field: 离线计算好的距离场 D(p_i) [Batch, N] (即 [2, 10000])
        :return: l_total, l_dwce, l_bls
        """
        # 0. 维度展平 (Flattening)
        # PyTorch 的交叉熵等函数通常喜欢接收 2D 的预测值和 1D 的标签
        B, N, C = logits.shape
        logits_flat = logits.view(-1, C)  # 变成 [20000, 5]
        targets_flat = targets.view(-1)  # 变成 [20000]
        dist_flat = distance_field.view(-1)  # 变成 [20000]

        # ==============================================================
        # ==============================================================
        # 1. 上半路：距离加权交叉熵损失 (L_DWCE) 融入了类别权重！
        # ==============================================================
        # 把 class_weights 传给 weight 参数
        ce_loss_pointwise = F.cross_entropy(logits_flat, targets_flat,
                                            weight=self.class_weights,
                                            reduction='none')

        # 乘以微观距离场 D(p_i) 的指数衰减权重
        weights_dist = 1.0 + self.gamma * torch.exp(-dist_flat / self.sigma)
        l_dwce = torch.mean(weights_dist * ce_loss_pointwise)

        # ==============================================================
        # 2. 下半路：边界结构损失 (L_BLS) (真实的 Lovász-Softmax 实现)
        # ==============================================================
        mask = dist_flat < self.tau
        if mask.sum() > 0:
            # 提取掩码界定内的 logits 和 targets
            logits_masked = logits_flat[mask]
            targets_masked = targets_flat[mask]

            #Lovász-Softmax 的输入必须是经过 Softmax 激活的后验概率！
            probs_masked = F.softmax(logits_masked, dim=-1)

            #调用真正的 Lovász 连续扩展直接优化 IoU 边界
            l_bls = lovasz_softmax_flat(probs_masked, targets_masked, classes='present')
        else:
            l_bls = torch.tensor(0.0, device=logits.device, requires_grad=True)

        l_total = self.lambda_weight * l_dwce + (1.0 - self.lambda_weight) * l_bls
        return l_total, l_dwce, l_bls

