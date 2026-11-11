import jax
from jax.experimental import pallas as pl

def softmax_kernel(x_ref, o_ref):
    x = x_ref[...]
    o_ref[...] = jax.numpy.exp(x - jax.numpy.max(x))
