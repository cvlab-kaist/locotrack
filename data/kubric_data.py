import functools
import tensorflow_datasets as tfds
from kubric.challenges.point_tracking.dataset import add_tracks


def create_point_tracking_dataset(
    data_dir=None,
    train_size=(256, 256),
    shuffle_buffer_size=256,
    split='train',
    batch_dims=tuple(),
    repeat=True,
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=25,
    max_sampled_frac=0.1,
    num_parallel_point_extraction_calls=16,
    snap_to_occluder=False,
    **kwargs):
  """Construct a dataset for point tracking using Kubric.

  Args:
    train_size: Tuple of 2 ints. Cropped output will be at this resolution
    shuffle_buffer_size: Int. Size of the shuffle buffer
    split: Which split to construct from Kubric.  Can be 'train' or
      'validation'.
    batch_dims: Sequence of ints. Add multiple examples into a batch of this
      shape.
    repeat: Bool. whether to repeat the dataset.
    vflip: Bool. whether to vertically flip the dataset to test generalization.
    random_crop: Bool. whether to randomly crop videos
    tracks_to_sample: Int. Total number of tracks to sample per video.
    sampling_stride: Int. For efficiency, query points are sampled from a
      random grid of this stride.
    max_seg_id: Int. The maxium segment id in the video.  Note the size of
      the to graph is proportional to this number, so prefer small values.
    max_sampled_frac: Float. The maximum fraction of points to sample from each
      object, out of all points that lie on the sampling grid.
    num_parallel_point_extraction_calls: Int. The num_parallel_calls for the
      map function for point extraction.
    snap_to_occluder: If true, query points within 1 pixel of occlusion 
      boundaries will track the occluding surface rather than the background.
      This results in models which are biased to track foreground objects
      instead of background.  Whether this is desirable depends on downstream
      applications.
    **kwargs: additional args to pass to tfds.load.

  Returns:
    The dataset generator.
  """
  ds = tfds.load(
      'panning_movi_e/256x256',
      data_dir=data_dir,
      shuffle_files=shuffle_buffer_size is not None,
      **kwargs)

  ds = ds[split]
  if repeat:
    ds = ds.repeat()
  ds = ds.map(
      functools.partial(
          add_tracks,
          train_size=train_size,
          vflip=vflip,
          random_crop=random_crop,
          tracks_to_sample=tracks_to_sample,
          sampling_stride=sampling_stride,
          max_seg_id=max_seg_id,
          max_sampled_frac=max_sampled_frac,
          snap_to_occluder=snap_to_occluder),
      num_parallel_calls=num_parallel_point_extraction_calls)
  if shuffle_buffer_size is not None:
    ds = ds.shuffle(shuffle_buffer_size)

  for bs in batch_dims[::-1]:
    ds = ds.batch(bs)

  return ds

