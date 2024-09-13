# Copyright 2024 DeepMind Technologies Limited
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

"""TAPIR models definition."""

import functools
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, Dict

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from models import nets, utils
from models.cmdtop import CMDTop


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
        x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
        min_deg: int, the minimum (inclusive) degree of the encoding.
        max_deg: int, the maximum (exclusive) degree of the encoding.
        legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device)
    if legacy_posenc_order:
        xb = x[..., None, :] * scales[:, None]
        four_feat = torch.reshape(
            torch.sin(torch.stack([xb, xb + 0.5 * np.pi], dim=-2)),
            list(x.shape[:-1]) + [-1]
        )
    else:
        xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def get_relative_positions(seq_len, reverse=False, device='cuda'):
    x = torch.arange(seq_len, device=device)[None, :]
    y = torch.arange(seq_len, device=device)[:, None]
    return torch.tril(x - y) if not reverse else torch.triu(y - x)


def get_alibi_slope(num_heads, device='cuda'):
    x = (24) ** (1 / num_heads)
    return torch.tensor([1 / x ** (i + 1) for i in range(num_heads)], device=device, dtype=torch.float32).view(-1, 1, 1)


class MultiHeadAttention(nn.Module):
    """Multi-headed attention (MHA) module."""

    def __init__(self, num_heads, key_size, w_init_scale=None, w_init=None, with_bias=True, b_init=None, value_size=None, model_size=None):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_heads

        self.with_bias = with_bias

        self.query_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.key_proj = nn.Linear(num_heads * key_size, num_heads * key_size, bias=with_bias)
        self.value_proj = nn.Linear(num_heads * self.value_size, num_heads * self.value_size, bias=with_bias)
        self.final_proj = nn.Linear(num_heads * self.value_size, self.model_size, bias=with_bias)

    def forward(self, query, key, value, mask=None):
        batch_size, sequence_length, _ = query.size()

        query_heads = self._linear_projection(query, self.key_size, self.query_proj)  # [T', H, Q=K]
        key_heads = self._linear_projection(key, self.key_size, self.key_proj)  # [T, H, K]
        value_heads = self._linear_projection(value, self.value_size, self.value_proj)  # [T, H, V]

        device = query.device
        bias_forward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(sequence_length, device=device)
        bias_forward = bias_forward + torch.triu(torch.full_like(bias_forward, -1e9), diagonal=1)
        bias_backward = get_alibi_slope(self.num_heads // 2, device=device) * get_relative_positions(sequence_length, reverse=True, device=device)
        bias_backward = bias_backward + torch.tril(torch.full_like(bias_backward, -1e9), diagonal=-1)
        attn_bias = torch.cat([bias_forward, bias_backward], dim=0)

        attn = F.scaled_dot_product_attention(query_heads, key_heads, value_heads, attn_mask=attn_bias, scale=1 / np.sqrt(self.key_size))
        attn = attn.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, -1)

        return self.final_proj(attn)  # [T', D']

    def _linear_projection(self, x, head_size, proj_layer):
        y = proj_layer(x)
        batch_size, sequence_length, _= x.shape
        return y.reshape((batch_size, sequence_length, self.num_heads, head_size)).permute(0, 2, 1, 3)


class Transformer(nn.Module):
    """A transformer stack."""

    def __init__(self, num_heads, num_layers, attn_size, dropout_rate, widening_factor=4):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadAttention(num_heads, attn_size, model_size=attn_size * num_heads),
                'dense': nn.Sequential(
                    nn.Linear(attn_size * num_heads, widening_factor * attn_size * num_heads),
                    nn.GELU(),
                    nn.Linear(widening_factor * attn_size * num_heads, attn_size * num_heads)
                ),
                'layer_norm1': nn.LayerNorm(attn_size * num_heads),
                'layer_norm2': nn.LayerNorm(attn_size * num_heads)
            })
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(attn_size * num_heads)

    def forward(self, embeddings, mask=None):
        h = embeddings
        for layer in self.layers:
            h_norm = layer['layer_norm1'](h)
            h_attn = layer['attn'](h_norm, h_norm, h_norm, mask=mask)
            h_attn = F.dropout(h_attn, p=self.dropout_rate, training=self.training)
            h = h + h_attn

            h_norm = layer['layer_norm2'](h)
            h_dense = layer['dense'](h_norm)
            h_dense = F.dropout(h_dense, p=self.dropout_rate, training=self.training)
            h = h + h_dense

        return self.ln_out(h)


class PIPSTransformer(nn.Module):
    def __init__(self, input_channels, output_channels, dim=512, num_heads=8, num_layers=1):
        super(PIPSTransformer, self).__init__()
        self.dim = dim

        self.transformer = Transformer(
            num_heads=num_heads,
            num_layers=num_layers,
            attn_size=dim // num_heads,
            dropout_rate=0.,
            widening_factor=4,
        )
        self.input_proj = nn.Linear(input_channels, dim)
        self.output_proj = nn.Linear(dim, output_channels)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x, mask=None)
        return self.output_proj(x)


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

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  highest: Sequence[torch.Tensor]
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

  lowres: Sequence[torch.Tensor]
  hires: Sequence[torch.Tensor]
  highest: Sequence[torch.Tensor]
  lowres_supp: Sequence[torch.Tensor]
  hires_supp: Sequence[torch.Tensor]
  highest_supp: Sequence[torch.Tensor]
  resolutions: Sequence[Tuple[int, int]]


class LocoTrack(nn.Module):
  """TAPIR model."""

  def __init__(
      self,
      bilinear_interp_with_depthwise_conv: bool = False,
      num_pips_iter: int = 4,
      pyramid_level: int = 0,
      mixer_hidden_dim: int = 512,
      num_mixer_blocks: int = 12,
      mixer_kernel_shape: int = 3,
      patch_size: int = 7,
      softmax_temperature: float = 20.0,
      parallelize_query_extraction: bool = False,
      initial_resolution: Tuple[int, int] = (256, 256),
      blocks_per_group: Sequence[int] = (2, 2, 2, 2),
      feature_extractor_chunk_size: int = 256,
      extra_convs: bool = False,
      use_casual_conv: bool = False,
      model_size: str = 'base',
  ):
    super().__init__()

    if model_size == 'small':
      model_params = {
        'dim': 256,
        'num_heads': 4,
        'num_layers': 3,
      }
      cmdtop_params = {
        'in_channel': 49,
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
        'in_channel': 49,
        'out_channels': (64, 128, 128),
        'kernel_shapes': (3, 3, 2), 
        'strides': (2, 2, 2),
      }
    else:
      raise ValueError(f"Unknown model size '{model_size}'")

    self.highres_dim = 128
    self.lowres_dim = 256
    self.bilinear_interp_with_depthwise_conv = (
        bilinear_interp_with_depthwise_conv
    )
    self.parallelize_query_extraction = parallelize_query_extraction

    self.num_pips_iter = num_pips_iter
    self.pyramid_level = pyramid_level
    self.patch_size = patch_size
    self.softmax_temperature = softmax_temperature
    self.initial_resolution = tuple(initial_resolution)
    self.feature_extractor_chunk_size = feature_extractor_chunk_size
    self.num_mixer_blocks = num_mixer_blocks
    self.use_casual_conv = use_casual_conv

    highres_dim = 128
    lowres_dim = 256
    strides = (1, 2, 2, 1)
    blocks_per_group = (2, 2, 2, 2)
    channels_per_group = (64, highres_dim, 256, lowres_dim)
    use_projection = (True, True, True, True)

    self.resnet_torch = nets.ResNet(
        blocks_per_group=blocks_per_group,
        channels_per_group=channels_per_group,
        use_projection=use_projection,
        strides=strides,
    )

    self.torch_pips_mixer = PIPSTransformer(
      input_channels=854,
      output_channels=4 + self.highres_dim + self.lowres_dim,
      **model_params
    )
    
    self.cmdtop = nn.ModuleList([
      CMDTop(
        **cmdtop_params
      ) for _ in range(3)
    ])

    self.cost_conv = utils.Conv2dSamePadding(2, 1, 3, 1)
    self.occ_linear = nn.Linear(6, 2)

    if extra_convs:
      self.extra_convs = nets.ExtraConvs()
    else:
      self.extra_convs = None

  def forward(
      self,
      video: torch.Tensor,
      query_points: torch.Tensor,
      feature_grids: Optional[FeatureGrids] = None,
      is_training: bool = False,
      query_chunk_size: Optional[int] = 64,
      get_query_feats: bool = False,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> Mapping[str, torch.Tensor]:
    """Runs a forward pass of the model.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      query_points: The query points for which we compute tracks.
      is_training: Whether we are training.
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

    if feature_grids is None:
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

    p = self.num_pips_iter
    out = dict(
        occlusion=torch.mean(
            torch.stack(trajectories['occlusion'][p::p]), dim=0
        ),
        tracks=torch.mean(torch.stack(trajectories['tracks'][p::p]), dim=0),
        expected_dist=torch.mean(
            torch.stack(trajectories['expected_dist'][p::p]), dim=0
        ),
        unrefined_occlusion=trajectories['occlusion'][:-1],
        unrefined_tracks=trajectories['tracks'][:-1],
        unrefined_expected_dist=trajectories['expected_dist'][:-1],
    )

    return out

  def get_query_features(
      self,
      video: torch.Tensor,
      is_training: bool,
      query_points: torch.Tensor,
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
      if utils.is_same_res(curr_resolution, resolution):
        query_feats.append(query_feats[-1])
        hires_query_feats.append(hires_query_feats[-1])
        highest_query_feats.append(highest_query_feats[-1])
        query_supp.append(query_supp[-1])
        hires_query_supp.append(hires_query_supp[-1])
        highest_query_supp.append(highest_query_supp[-1])
        continue
      position_in_grid = utils.convert_grid_coordinates(
          query_points,
          shape[1:4],
          feature_grid[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_hires = utils.convert_grid_coordinates(
          query_points,
          shape[1:4],
          hires_feats[i].shape[1:4],
          coordinate_format='tyx',
      )
      position_in_grid_highest = utils.convert_grid_coordinates(
          query_points,
          shape[1:4],
          highest_feats[i].shape[1:4],
          coordinate_format='tyx',
      )

      support_size = 7
      ctxx, ctxy = torch.meshgrid(
        torch.arange(-(support_size // 2), support_size // 2 + 1), 
        torch.arange(-(support_size // 2), support_size // 2 + 1),
        indexing='xy',
      )
      ctx = torch.stack([torch.zeros_like(ctxy), ctxy, ctxx], axis=-1)
      ctx = torch.reshape(ctx, [-1, 3]).to(video.device) # s*s 3

      position_support = position_in_grid[..., None, :] + ctx[None, None, ...] # b n s*s 3
      position_support = rearrange(position_support, 'b n s c -> b (n s) c')
      interp_supp = utils.map_coordinates_3d(
          feature_grid[i], position_support
      )
      interp_supp = rearrange(interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)

      position_support_hires = position_in_grid_hires[..., None, :] + ctx[None, None, ...]
      position_support_hires = rearrange(position_support_hires, 'b n s c -> b (n s) c')
      hires_interp_supp = utils.map_coordinates_3d(
          hires_feats[i], position_support_hires
      )
      hires_interp_supp = rearrange(hires_interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)

      position_support_highest = position_in_grid_highest[..., None, :] + ctx[None, None, ...]
      position_support_highest = rearrange(position_support_highest, 'b n s c -> b (n s) c')
      highest_interp_supp = utils.map_coordinates_3d(
          highest_feats[i], position_support_highest
      )
      highest_interp_supp = rearrange(highest_interp_supp, 'b (n h w) c -> b n h w c', h=support_size, w=support_size)

      interp_features = interp_supp[..., support_size // 2, support_size // 2, :]
      hires_interp = hires_interp_supp[..., support_size // 2, support_size // 2, :]
      highest_interp = highest_interp_supp[..., support_size // 2, support_size // 2, :]

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

  def get_feature_grids(
      self,
      video: torch.Tensor,
      is_training: Optional[bool] = False,
      refinement_resolutions: Optional[List[Tuple[int, int]]] = None,
  ) -> FeatureGrids:
    """Computes feature grids.

    Args:
      video: A 5-D tensor representing a batch of sequences of images.
      is_training: Whether we are training.
      refinement_resolutions: A list of (height, width) tuples. Refinement will
        be repeated at each specified resolution, to achieve high accuracy on
        resolutions higher than what TAPIR was trained on. If None, reasonable
        refinement resolutions will be inferred from the input video size.

    Returns:
      A FeatureGrids object containing the required features for every
      required resolution. Note that there will be one more feature grid
      than there are refinement_resolutions, because there is always a
      feature grid computed for TAP-Net initialization.
    """
    del is_training
    if refinement_resolutions is None:
      refinement_resolutions = utils.generate_default_resolutions(
          video.shape[2:4], self.initial_resolution
      )

    all_required_resolutions = []
    all_required_resolutions.extend(refinement_resolutions)

    feature_grid = []
    hires_feats = []
    highest_feats = []
    resize_im_shape = []
    curr_resolution = (-1, -1)

    latent = None
    hires = None
    video_resize = None
    for resolution in all_required_resolutions:
      if resolution[0] % 8 != 0 or resolution[1] % 8 != 0:
        raise ValueError('Image resolution must be a multiple of 8.')

      if not utils.is_same_res(curr_resolution, resolution):
        if utils.is_same_res(curr_resolution, video.shape[-3:-1]):
          video_resize = video
        else:
          video_resize = utils.bilinear(video, resolution)

        curr_resolution = resolution
        n, f, h, w, c = video_resize.shape
        video_resize = video_resize.view(n*f, h, w, c).permute(0, 3, 1, 2)

        if self.feature_extractor_chunk_size > 0:
          latent_list = []
          hires_list = []
          highest_list = []
          chunk_size = self.feature_extractor_chunk_size
          for start_idx in range(0, video_resize.shape[0], chunk_size):
            video_chunk = video_resize[start_idx:start_idx + chunk_size]
            resnet_out = self.resnet_torch(video_chunk)

            u3 = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
            latent_list.append(u3)
            u1 = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)
            hires_list.append(u1)
            u0 = resnet_out['resnet_unit_0'].permute(0, 2, 3, 1)
            highest_list.append(u0)

          latent = torch.cat(latent_list, dim=0)
          hires = torch.cat(hires_list, dim=0)
          highest = torch.cat(highest_list, dim=0)

        else:
          resnet_out = self.resnet_torch(video_resize)
          latent = resnet_out['resnet_unit_3'].permute(0, 2, 3, 1)
          hires = resnet_out['resnet_unit_1'].permute(0, 2, 3, 1)
          highest = resnet_out['resnet_unit_0'].permute(0, 2, 3, 1)

        if self.extra_convs:
          latent = self.extra_convs(latent)

        latent = latent / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(latent), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=latent.device),
            )
        )
        hires = hires / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(hires), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=hires.device),
            )
        )
        highest = highest / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(highest), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=highest.device),
            )
        )

        latent = latent.view(n, f, *latent.shape[1:])
        hires = hires.view(n, f, *hires.shape[1:])
        highest = highest.view(n, f, *highest.shape[1:])

      feature_grid.append(latent)
      hires_feats.append(hires)
      highest_feats.append(highest)
      resize_im_shape.append(video_resize.shape[2:4])

    return FeatureGrids(
        tuple(feature_grid), tuple(hires_feats), tuple(highest_feats), tuple(resize_im_shape)
    )

  def estimate_trajectories(
      self,
      video_size: Tuple[int, int],
      is_training: bool,
      feature_grids: FeatureGrids,
      query_features: QueryFeatures,
      query_points_in_video: Optional[torch.Tensor],
      query_chunk_size: Optional[int] = None,
      causal_context: Optional[Dict[str, torch.Tensor]] = None,
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
      causal_context: If provided, a dict of causal context to use for
        refinement.
      get_causal_context: If True, return causal context in the output.

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
    del is_training

    def train2orig(x):
      return utils.convert_grid_coordinates(
          x,
          self.initial_resolution[::-1],
          video_size[::-1],
          coordinate_format='xy',
      )

    occ_iters = []
    pts_iters = []
    expd_iters = []
    new_causal_context = []
    num_iters = self.num_pips_iter
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

    for ch in range(0, num_queries, query_chunk_size):
      chunk = query_features.lowres[0][:, ch:ch + query_chunk_size]
      chunk_hires = query_features.hires[0][:, ch:ch + query_chunk_size]

      if query_points_in_video is not None:
        infer_query_points = query_points_in_video[
            :, ch : ch + query_chunk_size
        ]
        num_frames = feature_grids.lowres[0].shape[1]
        infer_query_points = utils.convert_grid_coordinates(
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
        feature_level = -1
        queries = [
            query_features.hires[feature_level][:, ch:ch + query_chunk_size],
            query_features.lowres[feature_level][:, ch:ch + query_chunk_size],
            query_features.highest[feature_level][:, ch:ch + query_chunk_size],
        ]
        supports = [
            query_features.hires_supp[feature_level][:, ch:ch + query_chunk_size],
            query_features.lowres_supp[feature_level][:, ch:ch + query_chunk_size],
            query_features.highest_supp[feature_level][:, ch:ch + query_chunk_size],
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
              F.avg_pool3d(
                  pyramid[-1],
                  kernel_size=(2, 2, 1),
                  stride=(2, 2, 1),
                  padding=0,
              )
          )

        refined = self.refine_pips(
            queries,
            supports,
            None,
            pyramid,
            points.detach(),
            occlusion.detach(),
            expected_dist.detach(),
            orig_hw=self.initial_resolution,
            last_iter=mixer_feats,
            mixer_iter=i,
            resize_hw=feature_grids.resolutions[feature_level],
            get_causal_context=get_causal_context,
            cost_volume=cost_volume
        )
        points, occlusion, expected_dist, mixer_feats, new_causal = refined
        pts_iters[i + 1].append(train2orig(points))
        occ_iters[i + 1].append(occlusion)
        expd_iters[i + 1].append(expected_dist)
        new_causal_context[i].append(new_causal)

        if (i + 1) % self.num_pips_iter == 0:
          mixer_feats = None
          expected_dist = expd_iters[0][-1]
          occlusion = occ_iters[0][-1]

    occlusion = []
    points = []
    expd = []
    for i, _ in enumerate(occ_iters):
      occlusion.append(torch.cat(occ_iters[i], dim=1))
      points.append(torch.cat(pts_iters[i], dim=1))
      expd.append(torch.cat(expd_iters[i], dim=1))

    out = dict(
        occlusion=occlusion,
        tracks=points,
        expected_dist=expd,
    )
    return out

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
      cost_volume=None,
  ):
    del frame_features
    del mixer_iter
    orig_h, orig_w = orig_hw
    resized_h, resized_w = resize_hw
    corrs_pyr = []
    assert len(target_feature) == len(pyramid)
    for pyridx, (query, supp, grid) in enumerate(zip(target_feature, support_feature, pyramid)):
      # note: interp needs [y,x]
      coords = utils.convert_grid_coordinates(
          pos_guess, (orig_w, orig_h), grid.shape[-2:-4:-1]
      )
      coords = torch.flip(coords, dims=(-1,))

      support_size = 7
      ctxx, ctxy = torch.meshgrid(
        torch.arange(-(support_size // 2), support_size // 2 + 1), 
        torch.arange(-(support_size // 2), support_size // 2 + 1),
        indexing='xy',
      )
      ctx = torch.stack([ctxy, ctxx], dim=-1)
      ctx = ctx.reshape(-1, 2).to(coords.device)
      coords2 = coords.unsqueeze(3) + ctx.unsqueeze(0).unsqueeze(0).unsqueeze(0)
      neighborhood = utils.map_coordinates_2d(grid, coords2)

      neighborhood = rearrange(neighborhood, 'b n t (h w) c -> b n t h w c', h=support_size, w=support_size)
      patches_input = torch.einsum('bnthwc,bnijc->bnthwij', neighborhood, supp)
      patches_input = rearrange(patches_input, 'b n t h w i j -> (b n t) h w i j')
      patches_emb = self.cmdtop[pyridx](patches_input)
      patches = rearrange(patches_emb, '(b n t) c -> b n t c', b=neighborhood.shape[0], n=neighborhood.shape[1])

      corrs_pyr.append(patches)
    corrs_pyr = torch.concatenate(corrs_pyr, dim=-1)

    corrs_chunked = corrs_pyr
    pos_guess_input = pos_guess
    occ_guess_input = occ_guess[..., None]
    expd_guess_input = expd_guess[..., None]

    # mlp_input is batch, num_points, num_chunks, frames_per_chunk, channels
    if last_iter is None:
      both_feature = torch.cat([target_feature[0], target_feature[1]], axis=-1)
      mlp_input_features = torch.tile(
          both_feature.unsqueeze(2), (1, 1, corrs_chunked.shape[-2], 1)
      )
    else:
      mlp_input_features = last_iter

    mlp_input_list = [
        occ_guess_input,
        expd_guess_input,
        corrs_chunked
    ]

    rel_pos_forward = F.pad(pos_guess_input[..., :-1, :] - pos_guess_input[..., 1:, :], (0, 0, 0, 1))
    rel_pos_backward = F.pad(pos_guess_input[..., 1:, :] - pos_guess_input[..., :-1, :], (0, 0, 1, 0))
    scale = torch.tensor([resized_w / orig_w, resized_h / orig_h]) / torch.tensor([orig_w, orig_h])
    scale = scale.to(pos_guess_input.device)
    rel_pos_forward = rel_pos_forward * scale
    rel_pos_backward = rel_pos_backward * scale
    rel_pos_emb_input = posenc(torch.cat([rel_pos_forward, rel_pos_backward], axis=-1), min_deg=0, max_deg=10) # batch, num_points, num_frames, 84
    mlp_input_list.append(rel_pos_emb_input)
    mlp_input = torch.cat(mlp_input_list, axis=-1)

    x = rearrange(mlp_input, 'b n f c -> (b n) f c')
    res = self.torch_pips_mixer(x)

    res = rearrange(res, '(b n) f c -> b n f c', b=mlp_input.shape[0])

    pos_update = utils.convert_grid_coordinates(
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

  def tracks_from_cost_volume(
      self,
      interp_feature: torch.Tensor,
      interp_feature_hires: torch.Tensor,
      feature_grid: torch.Tensor,
      feature_grid_hires: torch.Tensor,
      query_points: Optional[torch.Tensor],
      im_shp=None,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    cost_volume = torch.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature,
        feature_grid,
    )
    cost_volume_hires = torch.einsum(
        'bnc,bthwc->tbnhw',
        interp_feature_hires,
        feature_grid_hires,
    )

    shape = cost_volume.shape
    batch_size, num_points = cost_volume.shape[1:3]

    interp_cost = rearrange(cost_volume, 't b n h w -> (t b n) () h w')
    interp_cost = F.interpolate(interp_cost, cost_volume_hires.shape[3:], mode='bilinear', align_corners=False)
    interp_cost = rearrange(interp_cost, '(t b n) () h w -> t b n h w', b=batch_size, n=num_points)
    cost_volume_stack = torch.stack(
        [
          interp_cost,
          cost_volume_hires,
        ], dim=-1
    )
    pos = rearrange(cost_volume_stack, 't b n h w c -> (t b n) c h w')
    pos = self.cost_conv(pos)
    pos = rearrange(pos, '(t b n) () h w -> b n t h w', b=batch_size, n=num_points)
    
    pos_sm = pos.reshape(pos.size(0), pos.size(1), pos.size(2), -1)
    softmaxed = F.softmax(pos_sm * self.softmax_temperature, dim=-1)
    pos = softmaxed.view_as(pos)

    points = utils.heatmaps_to_points(pos, im_shp, query_points=query_points)

    occlusion = torch.cat(
      [
        torch.mean(cost_volume_stack, dim=(-2, -3)),
        torch.amax(cost_volume_stack, dim=(-2, -3)),
        torch.amin(cost_volume_stack, dim=(-2, -3)),
      ], dim=-1
    )
    occlusion = self.occ_linear(occlusion)
    expected_dist = rearrange(occlusion[..., 1:2], 't b n () -> b n t', t=shape[0])
    occlusion = rearrange(occlusion[..., 0:1], 't b n () -> b n t', t=shape[0])

    return points, occlusion, expected_dist, rearrange(cost_volume, 't b n h w -> b n t h w')

  def construct_initial_causal_state(self, num_points, num_resolutions=1):
    """Construct initial causal state."""
    value_shapes = {}
    for i in range(self.num_mixer_blocks):
      value_shapes[f'block_{i}_causal_1'] = (1, num_points, 2, 512)
      value_shapes[f'block_{i}_causal_2'] = (1, num_points, 2, 2048)
    fake_ret = {
        k: torch.zeros(v, dtype=torch.float32) for k, v in value_shapes.items()
    }
    return [fake_ret] * num_resolutions * 4
  
  def inference(
      self, 
      video : Union[np.ndarray, torch.Tensor],
      query_points : torch.Tensor, 
      query_chunk_size : int = 64,
      resolution : Tuple[int, int] = (256, 256),
      query_format : str = 'tyx',
    ) -> dict:
    """
    Run inference on LocoTrack model.
    Args:
      model: LocoTrack model
      video: np.ndarray or torch.Tensor, shape [batch, time, height, width, 3], normalized to [-1, 1]
        if np.ndarray, it will be converted to torch.Tensor
        if dtype is uint8, it will be converted to float32 and normalized to [-1, 1]
      query_points: torch.Tensor, shape [batch, num_points, 3]
      query_chunk_size: int, default 64
      resolution: Tuple[int, int], default (256, 256)
      query_format: str, default 'tyx', query points format
    Returns:
      dict with keys: 'tracks', 'occlusion'
    """
    assert video.shape[-1] == 3, f'video shape should be [batch, time, height, width, 3], got {video.shape}'
    device = next(self.parameters()).device

    # query_format is not tyx, then convert query_points to tyx
    query_shuffle_ind = [query_format.index(c) for c in 'tyx']
    query_points = query_points[..., query_shuffle_ind].to(device)
    
    if isinstance(video, np.ndarray):
      video = torch.from_numpy(video).to(device)
    
    if video.dtype == torch.uint8:
      video = video.float() / 255.0 * 2 - 1

    B, _, H, W, _ = video.shape
    if (H, W) != resolution:
      video = rearrange(video, 'b t h w c -> (b t) c h w')
      video = F.interpolate(video, resolution, mode='bilinear', align_corners=False)  
      video = rearrange(video, '(b t) c h w -> b t h w c', b=B)

      query_points = query_points.clone()
      query_points[..., 1] = query_points[..., 1] / H * resolution[0]
      query_points[..., 2] = query_points[..., 2] / W * resolution[1]

    out = self.forward(video, query_points, query_chunk_size=query_chunk_size)
    tracks, occlusion, expected_dist = out['tracks'], out['occlusion'], out['expected_dist']

    tracks = tracks * torch.tensor([W / resolution[1], H / resolution[0]], device=tracks.device)
    
    pred_occ = torch.sigmoid(occlusion)
    pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
    pred_occ = pred_occ > 0.5  # threshold

    return {
      'tracks': tracks,
      'occlusion': pred_occ,
    }
    

CHECKPOINT_LINK = {
    'small': 'https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt',
    'base': 'https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt',
}

def load_model(ckpt_path=None, model_size='base'):
  if ckpt_path is None:
    ckpt_link = CHECKPOINT_LINK[model_size]
    state_dict = torch.hub.load_state_dict_from_url(ckpt_link, map_location='cpu')['state_dict']
  else:
    state_dict = torch.load(ckpt_path)['state_dict']
  state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}

  model = LocoTrack(model_size=model_size)
  model.load_state_dict(state_dict)
  model.eval()

  return model
