import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

import cupy

class Accuracy_topk(function.Function):
    def __init__(self, top_k):
        self.top_k = top_k

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim >= 2,
            t_type.dtype == numpy.int32,
            t_type.ndim == 1,
            t_type.shape[0] == x_type.shape[0],
        )
        for i in range(2, x_type.ndim.eval()):
            type_check.expect(x_type.shape[i] == 1)

    def forward(self, inputs):
        # gpu not impremented
        xp = numpy
        y, t = inputs
        y = y.reshape(len(y), -1)  # flatten
        y = cupy.asnumpy(y)
        t = cupy.asnumpy(t)
        argsorted_y = xp.argsort(y)[:,-self.top_k:]

	return xp.asarray(xp.any(argsorted_y.T == t, axis=0).mean(dtype='f')),

def accuracy_topk(y, t, top_k):
    """Computes muticlass classification accuracy of the minibatch.

    Args:
        y (Variable): Variable holding a matrix whose (i, j)-th element
            indicates the score of the class j at the i-th example.
        t (Variable): Variable holding an int32 vector of groundtruth labels.
        top_k (int): The number for top_k accuracy.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return Accuracy_topk(top_k=top_k)(y, t)
