import math
import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.max_iters = max_iters
        self.power = power
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * (1 - self.last_epoch/self.max_iters)**self.power, self.min_lr)
                for base_lr in self.base_lrs]

# 添加CosineAnnealingWarmRestartsWithWarmup学习率调度器
class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule with warmup, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        warmup_steps (int): Number of warmup steps. Default: 0.
        warmup_lr (float): Initial learning rate for warmup. Default: 1e-6.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False,
                 warmup_steps=0, warmup_lr=1e-6):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr

        # 修复父类初始化参数传递
        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # 实现warmup
        if self.last_epoch < self.warmup_steps:
            # 线性warmup从warmup_lr到初始学习率
            factor = float(self.last_epoch + 1) / float(max(1, self.warmup_steps))
            return [self.warmup_lr + (base_lr - self.warmup_lr) * factor for base_lr in self.base_lrs]

        # 余弦退火部分
        adjusted_epoch = self.last_epoch - self.warmup_steps
        if adjusted_epoch <= 0:
            return [self.warmup_lr for _ in self.base_lrs]
        elif self.T_mult == 1:
            T_cur = adjusted_epoch % self.T_0
            T_i = self.T_0
        else:
            n = int(math.log((adjusted_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            T_cur = adjusted_epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
            T_i = self.T_0 * self.T_mult ** (n)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = epoch

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]