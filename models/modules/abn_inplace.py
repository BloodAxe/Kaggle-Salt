import torch
import torch.nn as nn
import torch.nn.functional as functional

from models.modules.abn import ABN

try:
    from queue import Queue
except ImportError:
    from Queue import Queue

from .functions import *


class InPlaceABN(ABN):
    """InPlace Activated Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        """Creates an InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABN, self).__init__(num_features, eps, momentum, affine, activation, slope)

    def forward(self, x):
        return inplace_abn(x, self.weight, self.bias, self.running_mean, self.running_var,
                           self.training, self.momentum, self.eps, self.activation, self.slope)


class InPlaceABNSync(ABN):
    """InPlace Activated Batch Normalization with cross-GPU synchronization

    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
    """

    def __init__(self, num_features, devices=None, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu",
                 slope=0.01):
        """Creates a synchronized, InPlace Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        devices : list of int or None
            IDs of the GPUs that will run the replicas of this module.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(InPlaceABNSync, self).__init__(num_features, eps, momentum, affine, activation, slope)
        self.devices = devices if devices else list(range(torch.cuda.device_count()))

        # Initialize queues
        self.worker_ids = self.devices[1:]
        self.master_queue = Queue(len(self.worker_ids))
        self.worker_queues = [Queue(1) for _ in self.worker_ids]

    def forward(self, x):
        if x.get_device() == self.devices[0]:
            # Master mode
            extra = {
                "is_master": True,
                "master_queue": self.master_queue,
                "worker_queues": self.worker_queues,
                "worker_ids": self.worker_ids
            }
        else:
            # Worker mode
            extra = {
                "is_master": False,
                "master_queue": self.master_queue,
                "worker_queue": self.worker_queues[self.worker_ids.index(x.get_device())]
            }

        return inplace_abn_sync(x, self.weight, self.bias, self.running_mean, self.running_var,
                                extra, self.training, self.momentum, self.eps, self.activation, self.slope)

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum},' \
              ' affine={affine}, devices={devices}, activation={activation}'
        if self.activation == "leaky_relu":
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)
