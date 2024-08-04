import jax
import jax.numpy as jnp

import haiku as hk
from einshape import jax_einshape as einshape

def conv_block(x, out_channels, kernel_shape, stride, name=None):
    x = hk.Conv2D(out_channels, kernel_shape=kernel_shape, stride=stride, padding='SAME', name=name)(x)
    x = hk.GroupNorm(out_channels // 16)(x)
    x = jax.nn.relu(x)
    return x

class CMDTop(hk.Module):
  def __init__(self, out_channels, kernel_shapes, strides, name=None):
    super().__init__(name=name)
    self.out_channels = out_channels
    self.kernel_shapes = kernel_shapes
    self.strides = strides

    self.conv = [
        hk.Sequential([
            hk.Conv2D(
                out_channels[i],
                kernel_shape=kernel_shapes[i],
                stride=strides[i],
                padding='SAME'
            ),
            hk.GroupNorm(out_channels[i] // 16),
            jax.nn.relu
        
        ]) for i in range(len(out_channels))
    ]

  def __call__(self, x):
    """
    x: (b, h, w, i, j)
    """
    out1 = einshape('bhwij->bhw(ij)', x)
    out2 = einshape('bhwij->bij(hw)', x)
    for i in range(len(self.out_channels)):
        out1 = self.conv[i](out1)
    
    for i in range(len(self.out_channels)):
        out2 = self.conv[i](out2)

    out1 = jnp.mean(out1, axis=(1, 2)) # (b, out_channels[-1])
    out2 = jnp.mean(out2, axis=(1, 2)) # (b, out_channels[-1])

    return jnp.concatenate([out1, out2], axis=-1) # (b, 2*out_channels[-1])