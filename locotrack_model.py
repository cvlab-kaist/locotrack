# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TAPIR model definition."""

import functools
import dataclasses
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import chex
from einshape import jax_einshape as einshape
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from locotrack.models import resnet
from locotrack.utils import model_utils
from locotrack.utils import transforms

from locotrack.models.cmdtop import CMDTop


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
  """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

  Instead of computing [sin(x), cos(x)], we use the trig identity
  cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

  Args:
    x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
    min_deg: int, the minimum (inclusive) degree of the encoding.
    max_deg: int, the maximum (exclusive) degree of the encoding.
    legacy_posenc_order: bool, keep the same ordering as the original tf code.

  Returns:
    encoded: jnp.ndarray, encoded variables.
  """
  if min_deg == max_deg:
    return x
  scales = jnp.array([2**i for i in range(min_deg, max_deg)])
  if legacy_posenc_order:
    xb = x[Ellipsis, None, :] * scales[:, None]
    four_feat = jnp.reshape(
        jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)),
        list(x.shape[:-1]) + [-1])
  else:
    xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
  return jnp.concatenate([x] + [four_feat], axis=-1)

def get_relative_positions(seq_len: int, reverse=False):
    x = jnp.arange(seq_len)[None, :]
    y = jnp.arange(seq_len)[:, None]
    return jnp.tril(x - y) if not reverse else jnp.triu(y - x)

def get_alibi_slope(num_heads):
    x = (24) ** (1 / num_heads)
    return jnp.array([1 / x ** (i + 1) for i in range(num_heads)]).reshape(-1, 1, 1)

def _layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to `x` with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


class MultiHeadAttention(hk.Module):
  """Multi-headed attention (MHA) module.

  This module is intended for attending over sequences of vectors.

  Rough sketch:
  - Compute keys (K), queries (Q), and values (V) as projections of inputs.
  - Attention weights are computed as W = softmax(QK^T / sqrt(key_size)).
  - Output is another projection of WV^T.

  For more detail, see the original Transformer paper:
    "Attention is all you need" https://arxiv.org/abs/1706.03762.

  Glossary of shapes:
  - T: Sequence length.
  - D: Vector (embedding) size.
  - H: Number of attention heads.
  """

  def __init__(
      self,
      num_heads: int,
      key_size: int,
      # TODO(b/240019186): Remove `w_init_scale`.
      w_init_scale: Optional[float] = None,
      *,
      w_init: Optional[hk.initializers.Initializer] = None,
      with_bias: bool = True,
      b_init: Optional[hk.initializers.Initializer] = None,
      value_size: Optional[int] = None,
      model_size: Optional[int] = None,
      name: Optional[str] = None,
  ):
    """Initialises the module.

    Args:
      num_heads: Number of independent attention heads (H).
      key_size: The size of keys (K) and queries used for attention.
      w_init_scale: DEPRECATED. Please use w_init instead.
      w_init: Initialiser for weights in the linear map. Once `w_init_scale` is
        fully deprecated `w_init` will become mandatory. Until then it has a
        default value of `None` for backwards compatability.
      with_bias: Whether to add a bias when computing various linear
        projections.
      b_init: Optional initializer for bias. By default, zero.
      value_size: Optional size of the value projection (V). If None, defaults
        to the key size (K).
      model_size: Optional size of the output embedding (D'). If None, defaults
        to the key size multiplied by the number of heads (K * H).
      name: Optional name for this module.
    """
    super().__init__(name=name)
    self.num_heads = num_heads
    self.key_size = key_size
    self.value_size = value_size or key_size
    self.model_size = model_size or key_size * num_heads

    # Backwards-compatibility for w_init_scale.
    # if w_init_scale is not None:
    #   warnings.warn(
    #       "w_init_scale is deprecated; please pass an explicit weight "
    #       "initialiser instead.", DeprecationWarning)
    if w_init and w_init_scale:
      raise ValueError("Please provide only `w_init`, not `w_init_scale`.")
    if w_init is None and w_init_scale is None:
      raise ValueError("Please provide a weight initializer: `w_init`. "
                       "`w_init` will become mandatory once `w_init_scale` is "
                       "fully deprecated.")
    if w_init is None:
      w_init = hk.initializers.VarianceScaling(w_init_scale)
    self.w_init = w_init
    self.with_bias = with_bias
    self.b_init = b_init

  def __call__(
      self,
      query: jax.Array,
      key: jax.Array,
      value: jax.Array,
      mask: Optional[jax.Array] = None,
  ) -> jax.Array:
    """Computes (optionally masked) MHA with queries, keys & values.

    This module broadcasts over zero or more 'batch-like' leading dimensions.

    Args:
      query: Embeddings sequence used to compute queries; shape [..., T', D_q].
      key: Embeddings sequence used to compute keys; shape [..., T, D_k].
      value: Embeddings sequence used to compute values; shape [..., T, D_v].
      mask: Optional mask applied to attention weights; shape [..., H=1, T', T].

    Returns:
      A new sequence of embeddings, consisting of a projection of the
        attention-weighted value projections; shape [..., T', D'].
    """

    # In shape hints below, we suppress the leading dims [...] for brevity.
    # Hence e.g. [A, B] should be read in every case as [..., A, B].
    *leading_dims, sequence_length, _ = query.shape
    projection = self._linear_projection

    # Compute key/query/values (overload K/Q/V to denote the respective sizes).
    query_heads = projection(query, self.key_size, "query")  # [T', H, Q=K]
    key_heads = projection(key, self.key_size, "key")  # [T, H, K]
    value_heads = projection(value, self.value_size, "value")  # [T, H, V]

    bias_forward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(sequence_length)
    bias_forward = bias_forward + jnp.triu(jnp.full_like(bias_forward, -1e9), k=1)
    bias_backward = get_alibi_slope(self.num_heads // 2) * get_relative_positions(sequence_length, reverse=True)
    bias_backward = bias_backward + jnp.tril(jnp.full_like(bias_backward, -1e9), k=-1)
    attn_bias = jnp.concatenate([bias_forward, bias_backward], axis=0)

    # Compute attention weights.
    attn_logits = jnp.einsum("...thd,...Thd->...htT", query_heads, key_heads)
    attn_logits = attn_logits / np.sqrt(self.key_size).astype(key.dtype) + attn_bias
    if mask is not None:
      if mask.ndim != attn_logits.ndim:
        raise ValueError(
            f"Mask dimensionality {mask.ndim} must match logits dimensionality "
            f"{attn_logits.ndim}."
        )
      attn_logits = jnp.where(mask, attn_logits, -1e30)
    attn_weights = jax.nn.softmax(attn_logits)  # [H, T', T]

    # Weight the values by the attention and flatten the head vectors.
    attn = jnp.einsum("...htT,...Thd->...thd", attn_weights, value_heads)
    attn = jnp.reshape(attn, (*leading_dims, sequence_length, -1))  # [T', H*V]

    # Apply another projection to get the final embeddings.
    final_projection = hk.Linear(self.model_size, w_init=self.w_init,
                                 with_bias=self.with_bias, b_init=self.b_init)
    return final_projection(attn)  # [T', D']

  @hk.transparent
  def _linear_projection(
      self,
      x: jax.Array,
      head_size: int,
      name: Optional[str] = None,
  ) -> jax.Array:
    y = hk.Linear(self.num_heads * head_size, w_init=self.w_init,
                  with_bias=self.with_bias, b_init=self.b_init, name=name)(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, self.num_heads, head_size))

@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  num_heads: int  # Number of attention heads.
  num_layers: int  # Number of transformer (attention + MLP) layers to stack.
  attn_size: int  # Size of the attention (key, query, value) vectors.
  dropout_rate: float  # Probability with which to apply dropout.
  widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
  name: Optional[str] = None  # Optional identifier for the module.

  def __call__(
      self,
      embeddings: jax.Array,  # [B, T, D]
      mask: Optional[jax.Array] = None,
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""

    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    _, seq_len, model_size = embeddings.shape

    # Compute causal mask for autoregressive sequence modelling.
    # mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
    # causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
    # mask = mask * causal_mask  # [B, H=1, T, T]

    h = embeddings
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = MultiHeadAttention(
          num_heads=self.num_heads,
          key_size=self.attn_size,
          model_size=model_size,
          w_init=initializer,
      )
      h_norm = _layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential([
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
      ])
      h_norm = _layer_norm(h)
      h_dense = dense_block(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
      h = h + h_dense

    return _layer_norm(h)


class PIPSTransformer(hk.Module):
  def __init__(
      self,
      output_channels: int,
      dim: int = 512,
      num_heads: int = 8,
      num_layers: int = 1,
      name='pips_transformer',
  ):
    super().__init__(name=name)
    self.dim = dim

    self.transformer = Transformer(
      num_heads=num_heads,
      num_layers=num_layers,
      attn_size=dim // num_heads,
      dropout_rate=0.,
      widening_factor=4,
      name='transformer'
    )
    self._output_channels = output_channels

  def __call__(
      self,
      x: chex.Array,
  ):
    x = hk.Linear(self.dim)(x)
    x = self.transformer(x, mask=None)
    return hk.Linear(self._output_channels)(x)
  

def layernorm(x):
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=False)(x)


def depthwise_conv_residual(
    x: chex.Array,
    kernel_shape: int = 3,
    use_causal_conv: bool = False,
    causal_context: Optional[Mapping[str, chex.Array]] = None,
    get_causal_context: bool = False,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
  """First mlp in mixer."""
  cur_name = hk.experimental.current_name()
  name1 = cur_name + '_causal_1'
  if causal_context is not None:
    x = jnp.concatenate([causal_context[name1], x], axis=-2)
    # Because we're adding extra frames, the output of this convolution will
    # also have extra (invalid) frames that need to be removed afterward.
    num_extra = causal_context[name1].shape[-2]
  new_causal_context = {}
  if get_causal_context:
    # Keep only as many frames of x as needed for future frames.  This may be up
    # to (kernel_shape - 1) frames.
    new_causal_context[name1] = x[..., -(kernel_shape - 1) :, :]
  x = hk.DepthwiseConv1D(
      channel_multiplier=4,
      kernel_shape=kernel_shape,
      data_format='NWC',
      stride=1,
      padding=[(kernel_shape - 1, 0)] if use_causal_conv else 'SAME',
      name='mlp1_up',
  )(x)
  x = jax.nn.gelu(x)
  name2 = cur_name + '_causal_2'
  if causal_context is not None:
    x = jnp.concatenate([causal_context[name2], x[..., num_extra:, :]], axis=-2)
    num_extra = causal_context[name2].shape[-2]
  if get_causal_context:
    new_causal_context[name2] = x[..., -(kernel_shape - 1) :, :]

  x = hk.DepthwiseConv1D(
      channel_multiplier=1,
      kernel_shape=kernel_shape,
      data_format='NWC',
      stride=1,
      padding=[(kernel_shape - 1, 0)] if use_causal_conv else 'SAME',
      name='mlp1_up',
  )(x)
  if causal_context is not None:
    x = x[..., num_extra:, :]

  return (
      x[..., 0::4] + x[..., 1::4] + x[..., 2::4] + x[..., 3::4],
      new_causal_context,
  )


def conv_channels_mixer(x):
  """Second mlp in mixer."""
  in_channels = x.shape[-1]
  x = hk.Linear(in_channels * 4, name='mlp2_up')(x)
  x = jax.nn.gelu(x)
  x = hk.Linear(in_channels, name='mlp2_down')(x)
  return x


class PIPsConvBlock(hk.Module):
  """Transformer block (mha and ffw)."""

  def __init__(self, name='block', kernel_shape=3, use_causal_conv=False):
    super().__init__(name=name)
    self.kernel_shape = kernel_shape
    self.use_causal_conv = use_causal_conv

  def __call__(self, x, causal_context=None, get_causal_context=False):
    to_skip = x
    x = layernorm(x)
    x, new_causal_context = depthwise_conv_residual(
        x,
        self.kernel_shape,
        self.use_causal_conv,
        causal_context,
        get_causal_context,
    )
    x = x + to_skip
    to_skip = x
    x = layernorm(x)
    x = conv_channels_mixer(x)
    x = x + to_skip
    return x, new_causal_context


class PIPSMLPMixer(hk.Module):
  """Depthwise-conv version of PIPs's MLP Mixer."""

  def __init__(
      self,
      output_channels,
      hidden_dim=512,
      num_blocks=12,
      kernel_shape=3,
      use_causal_conv=False,
      name='pips_mlp_mixer',
  ):
    super().__init__(name=name)
    self._output_channels = output_channels
    self.hidden_dim = hidden_dim
    self._num_blocks = num_blocks
    self.kernel_shape = kernel_shape
    self.use_causal_conv = use_causal_conv

  def __call__(self, x, causal_context=None, get_causal_context=False):
    x = hk.Linear(self.hidden_dim)(x)
    all_causal_context = {}
    for _ in range(self._num_blocks):
      x, new_causal_context = PIPsConvBlock(
          kernel_shape=self.kernel_shape, use_causal_conv=self.use_causal_conv
      )(x, causal_context, get_causal_context)
      if get_causal_context:
        all_causal_context.update(new_causal_context)
    x = layernorm(x)
    return hk.Linear(self._output_channels)(x), all_causal_context


def construct_patch_kernel(pos, grid_size, patch_size=7):
  """A conv kernel that performs bilinear interpolation for a point."""
  # pos is n-by-2, [y,x]
  # grid_size is [heigh,width]
  # result is [1,n,kernel_height,kernel_width]
  pos = pos + (patch_size) / 2 - 1

  def gen_bump(pos, num):
    # pos is shape [n]
    # result is shape [n,num]
    res = jnp.arange(num)
    return jnp.maximum(
        0, 1 - jnp.abs(res[jnp.newaxis, :] - pos[:, jnp.newaxis])
    )

  x_bump = gen_bump(pos[:, 1], grid_size[1] - patch_size + 1)
  y_bump = gen_bump(pos[:, 0], grid_size[0] - patch_size + 1)

  kernel = (
      x_bump[:, jnp.newaxis, jnp.newaxis, :]
      * y_bump[:, jnp.newaxis, :, jnp.newaxis]
  )
  return kernel


def extract_patch_depthwise_conv(pos, corrs, patch_size=7):
  """Use a depthwise conv to extract a patch via bilinear interpolation."""
  # pos is n-by-2, [y,x], raster coordinates
  # arr is [num_points, height, width]
  # result is [num_points, height, width]
  # add an extra batch axis because conv needs it
  corrs = jnp.pad(
      corrs,
      (
          (0, 0),
          (patch_size - 1, patch_size - 1),
          (patch_size - 1, patch_size - 1),
      ),
  )[jnp.newaxis, ...]
  kernel = construct_patch_kernel(pos, corrs.shape[2:4], patch_size)
  dim_nums = jax.lax.ConvDimensionNumbers(
      lhs_spec=(0, 1, 2, 3), rhs_spec=(0, 1, 2, 3), out_spec=(0, 1, 2, 3)
  )
  # the [0] gets rid of the extra batch axis
  res = jax.lax.conv_general_dilated(
      corrs,
      kernel,
      (1, 1),
      'VALID',
      (1, 1),
      (1, 1),
      dim_nums,
      feature_group_count=kernel.shape[0],
  )[0]
  return res


def is_same_res(r1, r2):
  """Test if two image resolutions are the same."""
  return all([x == y for x, y in zip(r1, r2)])


class FeatureGrids(NamedTuple):
  """Feature grids for a video, used to compute trajectories.

  These are per-frame outputs of the encoding resnet.

  Attributes:
    lowres: Low-resolution features, one for each resolution; 256 channels.
    hires: High-resolution features, one for each resolution; 64 channels.
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[chex.Array]
  hires: Sequence[chex.Array]
  highest: Sequence[chex.Array]
  resolutions: Sequence[Tuple[int, int]]


class QueryFeatures(NamedTuple):
  """Query features used to compute trajectories.

  These are sampled from the query frames and are a full descriptor of the
  tracked points. They can be acquired from a query image and then reused in a
  separate video.

  Attributes:
    lowres: Low-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 256]
    hires: High-resolution features, one for each resolution; each has shape
      [batch, num_query_points, 64]
    resolutions: Resolutions used for trajectory computation.  There will be one
      entry for the initialization, and then an entry for each PIPs refinement
      resolution.
  """

  lowres: Sequence[chex.Array]
  hires: Sequence[chex.Array]
  highest: Sequence[chex.Array]
  lowres_supp: Sequence[chex.Array]
  hires_supp: Sequence[chex.Array]
  highest_supp: Sequence[chex.Array]
  resolutions: Sequence[Tuple[int, int]]


class LocoTrack(hk.Module):
  """TAPIR model."""

  def __init__(
      self,
      bilinear_interp_with_depthwise_conv: bool = False,
      num_pips_iter: int = 4,
      pyramid_level: int = 1,
      mixer_hidden_dim: int = 512,
      num_mixer_blocks: int = 12,
      mixer_kernel_shape: int = 3,
      patch_size: int = 7,
      softmax_temperature: float = 20.0,
      use_causal_conv: bool = False,
      parallelize_query_extraction: bool = False,
      initial_resolution: Tuple[int, int] = (256, 256),
      blocks_per_group: Sequence[int] = (2, 2, 2, 2),
      model_size: str = 'base',
      name: str = 'locotrack',
  ):
    super().__init__(name=name)

    self.highres_dim = 128
    self.lowres_dim = 256

    self.resnet = resnet.ResNet(
        resnet_v2=True,
        normalization='instancenorm',
        strides=(1, 2, 2, 1),
        blocks_per_group=blocks_per_group,
        channels_per_group=(64, self.highres_dim, 256, self.lowres_dim),
        use_projection=(True, True, True, True),
        use_max_pool=False,
        name='resnet',
    )
    
    self.cost_conv = hk.Conv2D(
        1,
        [3, 3],
        name='cost_volume',
        stride=[1, 1],
    )
    self.occ_linear = hk.Linear(2, name='occlusion_out')
    # self.cost_volume_track_mods = {
    #   'cost_conv': hk.Conv2D(
    #       1,
    #       [3, 3],
    #       name='cost_volume',
    #       stride=[1, 1],
    #   ),
        # 'hid1': hk.Conv2D(
        #     16,
        #     [3, 3],
        #     name='cost_volume_regression_1',
        #     stride=[1, 1],
        # ),
        # 'hid2': hk.Conv2D(
        #     1,
        #     [3, 3],
        #     name='cost_volume_regression_2',
        #     stride=[1, 1],
        # ),
        # 'hid3': hk.Conv2D(
        #     32,
        #     [3, 3],
        #     name='cost_volume_occlusion_1',
        #     stride=[2, 2],
        # ),
        # 'hid4': hk.Linear(16, name='cost_volume_occlusion_2'),
        # 'occ_out': hk.Linear(2, name='occlusion_out'),
        # 'regression_hid': hk.Linear(128, name='regression_hid'),
        # 'regression_out': hk.Linear(2, name='regression_out'),
        # 'conv_stats_conv1': hk.Conv2D(
        #     256,
        #     [5, 5],
        #     name='conv_stats_conv1',
        #     stride=[1, 1],
        # ),
        # 'conv_stats_conv2': hk.Conv2D(
        #     256,
        #     [3, 3],
        #     name='conv_stats_conv2',
        #     stride=[1, 1],
        # ),
        # 'conv_stats_linear': hk.Linear(32, name='conv_stats_linear'),
    # }
    if model_size == 'small':
      model_params = {
        'dim': 256,
        'num_heads': 4,
        'num_layers': 3,
      }
      cmdtop_params = {
        'out_channels': (64, 128),
        'kernel_shapes': (5, 2), 
        'strides': (4, 2),
      }
    elif model_size == 'base':
      model_params = {
        'dim': 384,
        'num_heads': 6,
        'num_layers': 3,
      }
      cmdtop_params = {
        'out_channels': (64, 128, 128),
        'kernel_shapes': (3, 3, 2), 
        'strides': (2, 2, 2),
      }
    elif model_size == 'large':
      model_params = {
        'dim': 384,
        'num_heads': 6,
        'num_layers': 3,
      }
      cmdtop_params = {
        'out_channels': (128, 192, 192), 
        'kernel_shapes': (3, 3, 2), 
        'strides': (2, 2, 2),
      }
    
    self.pips_mixer = PIPSTransformer(
          4 + self.highres_dim + self.lowres_dim,
          **model_params
        )
    self.cmdtop = [
      CMDTop(
        **cmdtop_params
      ) for _ in range(3)
    ]

    self.bilinear_interp_with_depthwise_conv = (
        bilinear_interp_with_depthwise_conv
    )
    self.parallelize_query_extraction = parallelize_query_extraction

    self.num_pips_iter = num_pips_iter
    self.pyramid_level = pyramid_level
    self.patch_size = patch_size
    self.softmax_temperature = softmax_temperature
    self.initial_resolution = tuple(initial_resolution)

  def tracks_from_cost_volume(
      self,
      interp_feature: chex.Array,
      interp_feature_hires: chex.Array,
      feature_grid: chex.Array,
      feature_grid_hires: chex.Array,
      query_points: Optional[chex.Array],
      im_shp: Optional[chex.Shape] = None,
  ) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Converts features into tracks by computing a cost volume.

    The computed cost volume will have shape
      [batch, num_queries, time, height, width], which can be very
      memory intensive.

    Args:
      interp_feature: A tensor of features for each query point, of shape
        [batch, num_queries, channels, heads].
      feature_grid: A tensor of features for the video, of shape [batch, time,
        height, width, channels, heads].
      query_points: When computing tracks, we assume these points are given as
        ground truth and we reproduce them exactly.  This is a set of points of
        shape [batch, num_points, 3], where each entry is [t, y, x] in frame/
        raster coordinates.
      im_shp: The shape of the original image, i.e., [batch, num_frames, time,
        height, width, 3].

    Returns:
      A 2-tuple of the inferred points (of shape
        [batch, num_points, num_frames, 2] where each point is [x, y]) and
        inferred occlusion (of shape [batch, num_points, num_frames], where
        each is a logit where higher means occluded)
    """

    # mods = self.cost_volume_track_mods
    # Note: time is first axis to prevent the TPU from padding
    cost_volume = jnp.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature,
        feature_grid,
    )
    cost_volume_hires = jnp.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature_hires,
        feature_grid_hires,
    )
    batch_size, num_points = cost_volume.shape[1:3]

    cost_volume_stack = jnp.stack(
        [
          jax.image.resize(cost_volume, cost_volume_hires.shape, method='bilinear'),
          cost_volume_hires,
        ], axis=-1
    )
    pos = einshape('tbnhwc->(tbn)hwc', cost_volume_stack)
    pos = self.cost_conv(pos)
    pos = einshape('(tbn)hw1->bnthw', pos, b=batch_size, n=num_points)
    # batch_size, num_points = cost_volume.shape[1:3]

    # cost_volume_reshaped = einshape('tbnhw->(tbn)hw1', cost_volume)
    # cost_volume = einshape('tbnhw->bnthw', cost_volume)

    # occlusion = mods['hid1'](cost_volume_stack)
    # occlusion = jax.nn.relu(occlusion)

    # pos = mods['hid2'](occlusion)
    # pos_rshp = einshape('(tb)hw1->t(b)hw1', pos, t=shape[0])

    # pos = einshape('t(bn)hw1->bnthw', pos_rshp, b=batch_size, n=num_points)
    pos = jax.nn.softmax(pos * self.softmax_temperature, axis=(-2, -1))
    points = model_utils.heatmaps_to_points(
        pos, im_shp, query_points=query_points
    )

    occlusion = jnp.concatenate(
      [
        jnp.mean(cost_volume_stack, axis=(-2, -3)),
        jnp.max(cost_volume_stack, axis=(-2, -3)),
        jnp.min(cost_volume_stack, axis=(-2, -3)),
      ], axis=-1
    )
    occlusion = self.occ_linear(occlusion)
    expected_dist = einshape(
        'tbn1->bnt', occlusion[..., 1:2]
    )
    occlusion = einshape(
        'tbn1->bnt', occlusion[..., 0:1]
    )

    # occlusion = mods['hid3'](occlusion)
    # occlusion = jax.nn.relu(occlusion)
    # occlusion = jnp.mean(occlusion, axis=(-2, -3))
    # occlusion = mods['hid4'](occlusion)
    # occlusion = jax.nn.relu(occlusion)
    # occlusion = mods['occ_out'](occlusion)
    # expected_dist = einshape(
    #     '(tbn)1->bnt', occlusion[..., 1:2], n=shape[2], t=shape[0]
    # )
    # occlusion = einshape(
    #     '(tbn)1->bnt', occlusion[..., 0:1], n=shape[2], t=shape[0]
    # )
    return points, occlusion, expected_dist, einshape('tbnhw->bnthw', cost_volume)

  def refine_pips(
      self,
      target_feature,
      support_feature,
      frame_features,
      pyramid,
      pos_guess,
      occ_guess,
      expd_guess,
      orig_hw,
      last_iter=None,
      mixer_iter=0.0,
      resize_hw=None,
      causal_context=None,
      get_causal_context=False,
      cost_volume=None, # bnthw
  ):
    # frame_features is batch, num_frames, height, width, channels
    # target_features is batch, num_points, channels
    # pos_guess is batch, num_points, num_frames,2
    orig_h, orig_w = orig_hw
    resized_h, resized_w = resize_hw

    # cost_volume = einshape('bnthw->bnt(hw)', cost_volume)

    corrs_pyr = []
    assert len(target_feature) == len(pyramid)
    for pyridx, (query, supp, grid) in enumerate(zip(target_feature, support_feature, pyramid)):
      # note: interp needs [y,x]
      coords = transforms.convert_grid_coordinates(
          pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1]
      )[..., ::-1]
      last_iter_query = None
      if last_iter is not None:
        if pyridx == 0:
          last_iter_query = last_iter[..., : self.highres_dim]
        else:
          last_iter_query = last_iter[..., self.highres_dim :]
      if not self.bilinear_interp_with_depthwise_conv:
        # on CPU, gathers are cheap and matmuls are expensive
        support_size = 7
        ctxx, ctxy = jnp.meshgrid(
          jnp.arange(-(support_size // 2), support_size // 2 + 1), 
          jnp.arange(-(support_size // 2), support_size // 2 + 1)
        )
        ctx = jnp.stack([ctxy, ctxx], axis=-1)
        ctx = jnp.reshape(ctx, [-1, 2])
        coords2 = (
            coords[:, :, :, jnp.newaxis, :]
            + ctx[jnp.newaxis, jnp.newaxis, jnp.newaxis, ...]
        )
        # grid is batch, frames, height, width, channels
        # coords is batch, num_points, frames, spatial, x/y
        # neighborhood = batch, num_points, frames, patch_height, patch_width,
        # channels
        neighborhood = jax.vmap(  # across batch
            jax.vmap(  # across frames
                jax.vmap(  # across patch context size
                    jax.vmap(  # across channels
                        functools.partial(model_utils.interp, mode='constant'),
                        in_axes=(-1, None),
                        out_axes=-1,
                    ),
                    in_axes=(None, -2),
                    out_axes=-2,
                ),
                in_axes=(0, 1),
                out_axes=1,
            )
        )(grid, coords2)
        # s is spatial context size
        # if last_iter_query is None:
          # patches = jnp.einsum('bnfsc,bnc->bnfs', neighborhood, query) # s: 49
        neighborhood = einshape('bnt(hw)c->bnthwc', neighborhood, h=support_size, w=support_size)
        patches_input = jnp.einsum('bnthwc,bnijc->bnthwij', neighborhood, supp)
        patches_input = einshape('bnthwij->(bnt)hwij', patches_input)
        patches_emb = self.cmdtop[pyridx](patches_input)
        patches = einshape('(bnt)c->bntc', patches_emb, b=neighborhood.shape[0], n=neighborhood.shape[1])
      else:
        # on TPU, matmul is cheap and gather is expensive, so we rewrite
        # the interpolation with a depthwise conv.
        if last_iter_query is None:
          corrs = jnp.einsum('bfhwc,bnc->bnfhw', grid, query)
        else:
          corrs = jnp.einsum('bfhwc,bnfc->bnfhw', grid, last_iter_query)
        n = corrs.shape[1]
        # coords is bnfs2
        # patches is batch,n,frames,height,width
        # vmap across batch dimension (because we have different points across
        # the batch) and across the frame axis (this could potentially be rolled
        # into the depthwise conv)
        extract_patch_depthwise_conv_ = functools.partial(
            extract_patch_depthwise_conv, patch_size=self.patch_size
        )
        patches = jax.vmap(extract_patch_depthwise_conv_)(
            einshape('bnfc->b(nf)c', coords), einshape('bnfhw->b(nf)hw', corrs)
        )
        patches = einshape('b(nf)hw->bnf(hw)', patches, n=n)
      corrs_pyr.append(patches)
    corrs_pyr = jnp.concatenate(corrs_pyr, axis=-1)

    corrs_chunked = corrs_pyr
    pos_guess_input = pos_guess
    occ_guess_input = occ_guess[..., jnp.newaxis]
    expd_guess_input = expd_guess[..., jnp.newaxis]

    # mlp_input is batch, num_points, num_chunks, frames_per_chunk, channels
    if last_iter is None:
      both_feature = jnp.concatenate(
          [target_feature[0], target_feature[1]], axis=-1
      )
      mlp_input_features = jnp.tile(
          both_feature[:, :, jnp.newaxis, :],
          (1, 1) + corrs_chunked.shape[-2:-1] + (1,),
      )
    else:
      mlp_input_features = last_iter

    # pos_guess_input = jnp.zeros_like(pos_guess_input)
    # mixer_i = int(mixer_iter >= self.num_pips_iter // 2)
    mlp_input_list = [
      # pos_guess_input,
      occ_guess_input,
      expd_guess_input,
      # mlp_input_features,
      corrs_chunked,
    ]
    # if mixer_i == 0: # for first two iterations, add cost volume
    #   mlp_input_list.append(cost_volume)

    #   pos_normalized = pos_guess_input / jnp.array([orig_w, orig_h])
    #   pos_emb_input = posenc(pos_normalized, min_deg=0, max_deg=10) # batch, num_points, num_frames, 42
    #   mlp_input_list.append(pos_emb_input)
    # else:
      # pos_guess_input: batch, num_points, num_frames, 2
    rel_pos_forward = jnp.pad(pos_guess_input[..., :-1, :] - pos_guess_input[..., 1:, :], ((0, 0), (0, 0), (0, 1), (0, 0)))
    rel_pos_backward = jnp.pad(pos_guess_input[..., 1:, :] - pos_guess_input[..., :-1, :], ((0, 0), (0, 0), (1, 0), (0, 0)))
    scale = jnp.array([resized_w / orig_w, resized_h / orig_h]) / jnp.array([orig_w, orig_h])
    rel_pos_forward = rel_pos_forward * scale
    rel_pos_backward = rel_pos_backward * scale
    rel_pos_emb_input = posenc(jnp.concatenate([rel_pos_forward, rel_pos_backward], axis=-1), min_deg=0, max_deg=10) # batch, num_points, num_frames, 84
    mlp_input_list.append(rel_pos_emb_input)
    mlp_input = jnp.concatenate(mlp_input_list, axis=-1)
    
    x = einshape('bnfc->(bn)fc', mlp_input)
    if causal_context is not None:
      causal_context = jax.tree_map(
          lambda x: einshape('bn...->(bn)...', x), causal_context
      )

    # res, new_causal_context = self.pips_mixer[mixer_i](
    #     x, causal_context, get_causal_context
    # )
    res = self.pips_mixer(x)

    res = einshape('(bn)fc->bnfc', res, b=mlp_input.shape[0])
    if get_causal_context:
      new_causal_context = jax.tree_map(
          lambda x: einshape('(bn)...->bn...', x, b=mlp_input.shape[0]),
          new_causal_context,
      )

    pos_update = transforms.convert_grid_coordinates(
        res[..., :2],
        (resized_w, resized_h),
        (orig_w, orig_h),
    )
    return (
        pos_update + pos_guess,
        res[..., 2] + occ_guess,
        res[..., 3] + expd_guess,
        res[..., 4:] + (mlp_input_features if last_iter is None else last_iter),
        None,
    )

  def get_feature_grids(
      self,
      video: chex.Array,
      is_training: bool,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> FeatureGrids:
    """Computes feature grids.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A FeatureGrids object which contains the required features for every
        required resolution.  Note that there will be one more feature grid
        than there are refinement_resolutions, because there is always a
        feature grid computed for TAP-Net initialization.
    """

    if refinement_resolutions is None:
      refinement_resolutions = model_utils.generate_default_resolutions(
          video.shape[2:4], self.initial_resolution
      )

    all_required_resolutions = [self.initial_resolution]
    all_required_resolutions.extend(refinement_resolutions)

    feature_grid = []
    hires_feats = []
    highest_feats = []
    resize_im_shape = []
    curr_resolution = (-1, -1)
    for resolution in all_required_resolutions:
      if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
        raise ValueError('Image resolution must be a multiple of 8.')

      if not is_same_res(curr_resolution, resolution):
        curr_resolution = resolution
        if is_same_res(curr_resolution, video.shape[-3:-1]):
          video_resize = video
        else:
          video_resize = jax.image.resize(
              video, video.shape[0:2] + resolution + (3,), method='bilinear'
          )

        resnet_latent_list = []
        resnet_hires_list = []
        resnet_highest_list = []
        for i in range(0, video_resize.shape[1], 25):
          resnet_out = hk.BatchApply(self.resnet)(
              video_resize[:, i:min(i + 25, video_resize.shape[1]), ...], is_training
          )
          resnet_latent_list.append(resnet_out['resnet_unit_3'])
          resnet_hires_list.append(resnet_out['resnet_unit_1'])
          resnet_highest_list.append(resnet_out['resnet_unit_0'])
        
        latent = jnp.concatenate(resnet_latent_list, axis=1)
        hires = jnp.concatenate(resnet_hires_list, axis=1)
        highest = jnp.concatenate(resnet_highest_list, axis=1)
        latent = latent / jnp.sqrt(
            jnp.maximum(
                jnp.sum(jnp.square(latent), axis=-1, keepdims=True),
                1e-12,
            )
        )
        hires = hires / jnp.sqrt(
            jnp.maximum(
                jnp.sum(jnp.square(hires), axis=-1, keepdims=True),
                1e-12,
            )
        )
        highest = highest / jnp.sqrt(
            jnp.maximum(
                jnp.sum(jnp.square(highest), axis=-1, keepdims=True),
                1e-12,
            )
        )

      # feature_grid.append(jax.lax.stop_gradient(latent))
      # hires_feats.append(jax.lax.stop_gradient(hires))
      feature_grid.append(latent)
      hires_feats.append(hires)
      highest_feats.append(highest)
      resize_im_shape.append(video_resize.shape[2:4])

    return FeatureGrids(
        tuple(feature_grid), tuple(hires_feats), tuple(highest_feats), tuple(resize_im_shape)
    )

  def get_query_features(
      self,
      video: chex.Array,
      is_training: bool,
      query_points: chex.Array,
      feature_grids: Optional[FeatureGrids] = None,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> QueryFeatures:
    """Computes query features, which can be used for estimate_trajectories.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      feature_grids: If passed, we'll use these feature grids rather than
        computing new ones.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A QueryFeatures object which contains the required features for every
        required resolution.
    """

    if feature_grids is None:
      feature_grids = self.get_feature_grids(
          video,
          is_training=is_training,
          refinement_resolutions=refinement_resolutions,
      )

    feature_grid = feature_grids.lowres
    hires_feats = feature_grids.hires
    highest_feats = feature_grids.highest
    resize_im_shape = feature_grids.resolutions

    shape = video.shape

    # shape is [batch_size, time, height, width, channels]; conversion needs
    # [time, width, height]
    curr_resolution = (-1, -1)
    query_feats = []
    hires_query_feats = []
    highest_query_feats = []
    query_supp = []
    hires_query_supp = []
    highest_query_supp = []
    for i, resolution in enumerate(resize_im_shape):
      if is_same_res(curr_resolution, resolution):
        query_feats.append(query_feats[-1])
        hires_query_feats.append(hires_query_feats[-1])
        highest_query_feats.append(highest_query_feats[-1])
        query_supp.append(query_supp[-1])
        hires_query_supp.append(hires_query_supp[-1])
        highest_query_supp.append(highest_query_supp[-1])
        continue
      position_in_grid = transforms.convert_grid_coordinates(
          query_points,
          shape[1:4],
          feature_grid[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_hires = transforms.convert_grid_coordinates(
          query_points,
          shape[1:4],
          hires_feats[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_highest = transforms.convert_grid_coordinates(
          query_points,
          shape[1:4],
          highest_feats[i].shape[1:4],
          coordinate_format='tyx',
      )
      # if self.parallelize_query_extraction:
      #   # Extracting query features involves gathering features across frames;
      #   # a naive implementation will cause the model to do an all gather of
      #   # the full video feature tensor, which consumes lots of memory.
      #   # Therefore, we instead perform the x,y gather for every point on every
      #   # single frame, and then mask out the gathers on the incorrect frames.
      #   # This could be made more efficient by gathering exactly one query
      #   # feature per device (rather than per frame).
      #   #
      #   # interp_features is now [batch, time, num_points, features]
      #   interp_features = jax.vmap(
      #       jax.vmap(
      #           jax.vmap(
      #               model_utils.interp,
      #               in_axes=(2, None),
      #               out_axes=-1,
      #           ),
      #           in_axes=(0, None),
      #       )
      #   )(feature_grid[i], position_in_grid[..., 1:])
      #   # is_correct_frame is [batch, time, num_points]
      #   frame_id = jnp.array(jnp.round(position_in_grid[:, :, 0]), jnp.int32)
      #   is_correct_frame = jax.nn.one_hot(
      #       frame_id, feature_grid[i].shape[1], axis=1
      #   )
      #   interp_features = jnp.sum(
      #       interp_features * is_correct_frame[..., jnp.newaxis], axis=1
      #   )
      #   hires_interp = jax.vmap(
      #       jax.vmap(
      #           jax.vmap(
      #               model_utils.interp,
      #               in_axes=(2, None),
      #               out_axes=-1,
      #           ),
      #           in_axes=(0, None),
      #       )
      #   )(hires_feats[i], position_in_grid_hires[..., 1:])
      #   hires_interp = jnp.sum(
      #       hires_interp * is_correct_frame[..., jnp.newaxis], axis=1
      #   )
      # else:
      #   interp_features = jax.vmap(
      #       jax.vmap(
      #           model_utils.interp,
      #           in_axes=(3, None),
      #           out_axes=1,
      #       )
      #   )(feature_grid[i], position_in_grid)

      #   hires_interp = jax.vmap(
      #       jax.vmap(
      #           model_utils.interp,
      #           in_axes=(3, None),
      #           out_axes=1,
      #       )
      #   )(hires_feats[i], position_in_grid_hires)
      
      support_size = 7
      ctxx, ctxy = jnp.meshgrid(
        jnp.arange(-(support_size // 2), support_size // 2 + 1), 
        jnp.arange(-(support_size // 2), support_size // 2 + 1)
      )
      ctx = jnp.stack([jnp.zeros_like(ctxy), ctxy, ctxx], axis=-1)
      ctx = jnp.reshape(ctx, [-1, 3])
      position_support = position_in_grid[..., jnp.newaxis, :] + ctx[jnp.newaxis, jnp.newaxis, ...] # b n s*s 3
      interp_supp = jax.vmap(
        jax.vmap(
            jax.vmap(
                model_utils.interp,
                in_axes=(3, None), # in: thw, n3 -> thwc, n3 out: n->nc
                out_axes=1,
            ),
            in_axes=(None, 1), # in: thwc, ns3 
            out_axes=1,
        )
        )(feature_grid[i], position_support) # b n s*s c
      position_support_hires = position_in_grid_hires[..., jnp.newaxis, :] + ctx[jnp.newaxis, jnp.newaxis, ...]
      hires_interp_supp = jax.vmap(
        jax.vmap(
            jax.vmap(
                model_utils.interp,
                in_axes=(3, None), # in: thw, n3 -> thwc, n3 out: n->nc
                out_axes=1,
            ),
            in_axes=(None, 1), # in: thwc, ns3 
            out_axes=1,
        )
        )(hires_feats[i], position_support_hires) # b n s*s c
      position_support_highest = position_in_grid_highest[..., jnp.newaxis, :] + ctx[jnp.newaxis, jnp.newaxis, ...]
      highest_interp_supp = jax.vmap(
        jax.vmap(
            jax.vmap(
                model_utils.interp,
                in_axes=(3, None), # in: thw, n3 -> thwc, n3 out: n->nc
                out_axes=1,
            ),
            in_axes=(None, 1), # in: thwc, ns3 
            out_axes=1,
        )
        )(highest_feats[i], position_support_highest)

      interp_features = interp_supp[:, :, interp_supp.shape[2] // 2]
      hires_interp = hires_interp_supp[:, :, hires_interp_supp.shape[2] // 2]
      highest_interp = highest_interp_supp[:, :, highest_interp_supp.shape[2] // 2]

      interp_supp = einshape('bn(ij)c->bnijc', interp_supp, i=support_size, j=support_size)
      hires_interp_supp = einshape('bn(ij)c->bnijc', hires_interp_supp, i=support_size, j=support_size)
      highest_interp_supp = einshape('bn(ij)c->bnijc', highest_interp_supp, i=support_size, j=support_size)

      hires_query_feats.append(hires_interp)
      query_feats.append(interp_features)
      highest_query_feats.append(highest_interp)

      query_supp.append(interp_supp)
      hires_query_supp.append(hires_interp_supp)
      highest_query_supp.append(highest_interp_supp)

    return QueryFeatures(
        tuple(query_feats), tuple(hires_query_feats), tuple(highest_query_feats), 
        tuple(query_supp), tuple(hires_query_supp), tuple(highest_query_supp), tuple(resize_im_shape),
    )

  def estimate_trajectories(
      self,
      video_size: Tuple[int, int],
      is_training: bool,
      feature_grids: FeatureGrids,
      query_features: QueryFeatures,
      query_points_in_video: Optional[chex.Array],
      query_chunk_size: Optional[int] = None,
      causal_context: Optional[Sequence[Mapping[str, chex.Array]]] = None,
      get_causal_context: bool = False,
  ) -> Mapping[str, Any]:
    """Estimates trajectories given features for a video and query features.

    Args:
      video_size: A 2-tuple containing the original [height, width] of the
        video.  Predictions will be scaled with respect to this resolution.
      is_training: Whether we are training.
      feature_grids: a FeatureGrids object computed for the given video.
      query_features: a QueryFeatures object computed for the query points.
      query_points_in_video: If provided, assume that the query points come from
        the same video as feature_grids, and therefore constrain the resulting
        trajectories to (approximately) pass through them.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      causal_context: if running online, this will be features computed for each
        trajectory on earlier frames.  If None, no context is assumed.
      get_causal_context: if True, the output dict will also include features
        that can be passed as causal_context to future frames when running
        online.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
        causal_context: (if get_causal_context is True) a pytree which can
          be passed as causal_context to subsequent calls if running the model
          online.  In the current model, it is a list of dicts, one per
          PIPs refinement iteration, where for each dict the values are hidden
          units from the temporal depthwise conv layers, and the keys are names
          derived from Haiku's layer names.
    """

    def train2orig(x):
      return transforms.convert_grid_coordinates(
          x,
          self.initial_resolution[::-1],
          video_size[::-1],
          coordinate_format='xy',
      )

    occ_iters = []
    pts_iters = []
    expd_iters = []
    new_causal_context = []
    num_iters = self.num_pips_iter * (len(feature_grids.lowres) - 1)
    # This contains both middle step points and final step points.
    for _ in range(num_iters + 1):
      occ_iters.append([])
      pts_iters.append([])
      expd_iters.append([])
      new_causal_context.append([])
    del new_causal_context[-1]

    infer = functools.partial(
        self.tracks_from_cost_volume,
        im_shp=feature_grids.lowres[0].shape[0:2]
        + self.initial_resolution
        + (3,),
    )

    num_queries = query_features.lowres[0].shape[1]

    # Note: the permutation is required in order to randomize which tracks
    # get the stop_gradient.
    if causal_context is None:
      perm = jax.random.permutation(hk.next_rng_key(), num_queries)
    else:
      if is_training:
        # Need to handle permutation if we want to train with causal context.
        raise ValueError('Training with causal context is not supported.')
      perm = jnp.arange(num_queries, dtype=jnp.int32)
    inv_perm = jnp.zeros_like(perm)
    inv_perm = inv_perm.at[perm].set(jnp.arange(num_queries))

    for ch in range(0, num_queries, query_chunk_size):
      perm_chunk = perm[ch : ch + query_chunk_size]
      chunk = query_features.lowres[0][:, perm_chunk]
      chunk_hires = query_features.hires[0][:, perm_chunk]
      if causal_context is not None:
        cc_chunk = jax.tree_map(lambda x: x[:, perm_chunk], causal_context)  # pylint: disable=cell-var-from-loop
      if query_points_in_video is not None:
        infer_query_points = query_points_in_video[
            :, perm[ch : ch + query_chunk_size]
        ]
        num_frames = feature_grids.lowres[0].shape[1]
        infer_query_points = transforms.convert_grid_coordinates(
            infer_query_points,
            (num_frames,) + video_size,
            (num_frames,) + self.initial_resolution,
            coordinate_format='tyx',
        )
      else:
        infer_query_points = None
      points, occlusion, expected_dist, cost_volume = infer(
          chunk,
          chunk_hires,
          feature_grids.lowres[0],
          feature_grids.hires[0],
          infer_query_points,
      )
      pts_iters[0].append(train2orig(points))
      occ_iters[0].append(occlusion)
      expd_iters[0].append(expected_dist)

      mixer_feats = None

      for i in range(num_iters):
        feature_level = i // self.num_pips_iter + 1
        queries = [
            query_features.hires[feature_level][:, perm_chunk],
            query_features.lowres[feature_level][:, perm_chunk],
            query_features.highest[feature_level][:, perm_chunk],
        ]
        supports = [
            query_features.hires_supp[feature_level][:, perm_chunk],
            query_features.lowres_supp[feature_level][:, perm_chunk],
            query_features.highest_supp[feature_level][:, perm_chunk],
        ]
        for _ in range(self.pyramid_level):
          queries.append(queries[-1])
        pyramid = [
            feature_grids.hires[feature_level],
            feature_grids.lowres[feature_level],
            feature_grids.highest[feature_level],
        ]
        for _ in range(self.pyramid_level):
          pyramid.append(
              hk.avg_pool(
                  pyramid[-1], [1, 1, 2, 2, 1], [1, 1, 2, 2, 1], 'VALID'
              )
          )

        # Note: even when the pyramids are higher resolution, the points are
        # all scaled according to the original resolution.  This is because
        # the raw points are input into the model, and need to be scaled that
        # way.
        #
        # TODO(doersch): this should constrain the output to match the query
        # points.
        cc = cc_chunk[i] if causal_context is not None else None
        refined = self.refine_pips(
            queries,
            supports,
            None,
            pyramid,
            jax.lax.stop_gradient(points),
            jax.lax.stop_gradient(occlusion),
            jax.lax.stop_gradient(expected_dist),
            orig_hw=self.initial_resolution,
            last_iter=mixer_feats,
            mixer_iter=i,
            resize_hw=feature_grids.resolutions[feature_level],
            causal_context=cc,
            get_causal_context=get_causal_context,
            cost_volume=cost_volume
        )
        # if ch > 0:
        # refined = jax.lax.stop_gradient(refined)
        points = refined[0]
        occlusion = refined[1]
        expected_dist = refined[2]
        mixer_feats = refined[3]
        new_causal_context[i].append(refined[4])

        pts_iters[i + 1].append(train2orig(points))
        occ_iters[i + 1].append(occlusion)
        expd_iters[i + 1].append(expected_dist)

        if (i + 1) % self.num_pips_iter == 0:
          mixer_feats = None
          expected_dist = expd_iters[0][-1]
          occlusion = occ_iters[0][-1]

    occlusion = []
    points = []
    expd = []

    for i in range(len(occ_iters)):
      occlusion.append(jnp.concatenate(occ_iters[i], axis=1)[:, inv_perm])
      points.append(jnp.concatenate(pts_iters[i], axis=1)[:, inv_perm])
      expd.append(jnp.concatenate(expd_iters[i], axis=1)[:, inv_perm])

    for i in range(len(new_causal_context)):
      new_causal_context[i] = jax.tree_map(
          lambda *x: jnp.concatenate(x, axis=1)[:, inv_perm],
          *new_causal_context[i],
      )

    out = dict(
        occlusion=occlusion,
        tracks=points,
        expected_dist=expd,
    )
    if get_causal_context:
      out['causal_context'] = new_causal_context

    return out

  def __call__(
      self,
      video: chex.Array,
      is_training: bool,
      query_points: chex.Array,
      query_chunk_size: Optional[int] = None,
      get_query_feats: bool = False,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> Mapping[str, chex.Array]:
    """Runs a forward pass of the model.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      query_points: The query points for which we compute tracks.
      query_chunk_size: When computing cost volumes, break the queries into
        chunks of this size to save memory.
      get_query_feats: Return query features for other losses like contrastive.
        Not supported in the current version.
      refinement_resolutions: A list of (height, width) tuples.  Refinement will
        be repeated at each specified resolution, in order to achieve high
        accuracy on resolutions higher than what TAPIR was trained on. If None,
        reasonable refinement resolutions will be inferred from the input video
        size.

    Returns:
      A dict of outputs, including:
        occlusion: Occlusion logits, of shape [batch, num_queries, num_frames]
          where higher indicates more likely to be occluded.
        tracks: predicted point locations, of shape
          [batch, num_queries, num_frames, 2], where each point is [x, y]
          in raster coordinates
        expected_dist: uncertainty estimate logits, of shape
          [batch, num_queries, num_frames], where higher indicates more likely
          to be far from the correct answer.
    """
    if get_query_feats:
      raise ValueError('Get query feats not supported in TAPIR.')

    feature_grids = self.get_feature_grids(
        video,
        is_training,
        refinement_resolutions,
    )
    query_features = self.get_query_features(
        video,
        is_training,
        query_points,
        feature_grids,
        refinement_resolutions,
    )

    trajectories = self.estimate_trajectories(
        video.shape[-3:-1],
        is_training,
        feature_grids,
        query_features,
        query_points,
        query_chunk_size,
    )

    # The prediction is the average of the iterative refinement output across
    # every resolution.  At training time there's only one iteration, so we'll
    # just get the final refined output.  At test time, however, there will be
    # more.  The lowest resolution is at index self.num_pips_iter;
    # self.num_pips_iter iterations after that will be the next resolution,
    # and so on.
    p = self.num_pips_iter
    out = dict(
        occlusion=jnp.mean(jnp.stack(trajectories['occlusion'][p::p]), axis=0),
        tracks=jnp.mean(jnp.stack(trajectories['tracks'][p::p]), axis=0),
        expected_dist=jnp.mean(
            jnp.stack(trajectories['expected_dist'][p::p]), axis=0
        ),
        unrefined_occlusion=trajectories['occlusion'][:-1],
        unrefined_tracks=trajectories['tracks'][:-1],
        unrefined_expected_dist=trajectories['expected_dist'][:-1],
    )

    return out
