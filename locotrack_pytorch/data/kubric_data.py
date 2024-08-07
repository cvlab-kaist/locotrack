from typing import Mapping

import torch
import numpy as np

import functools
import tensorflow_datasets as tfds
import tensorflow as tf
import torch.distributed
from kubric.challenges.point_tracking.dataset import add_tracks


# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'


def default_color_augmentation_fn(
        inputs: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Standard color augmentation for videos.

    Args:
        inputs: A DatasetElement containing the item 'video' which will have
        augmentations applied to it.

    Returns:
        A DatasetElement with all the same data as the original, except that
        the video has augmentations applied.
    """
    zero_centering_image = True
    prob_color_augment = 0.8
    prob_color_drop = 0.2

    frames = inputs['video']
    if frames.dtype != tf.float32:
        raise ValueError('`frames` should be in float32.')

    def color_augment(video: tf.Tensor) -> tf.Tensor:
        """Do standard color augmentations."""
        # Note the same augmentation will be applied to all frames of the video.
        if zero_centering_image:
            video = 0.5 * (video + 1.0)
        video = tf.image.random_brightness(video, max_delta=32. / 255.)
        video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
        video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
        video = tf.image.random_hue(video, max_delta=0.2)
        video = tf.clip_by_value(video, 0.0, 1.0)
        if zero_centering_image:
            video = 2 * (video-0.5)
        return video

    def color_drop(video: tf.Tensor) -> tf.Tensor:
        video = tf.image.rgb_to_grayscale(video)
        video = tf.tile(video, [1, 1, 1, 1, 3])
        return video

    # Eventually applies color augmentation.
    coin_toss_color_augment = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    frames = tf.cond(
        pred=tf.less(coin_toss_color_augment,
                    tf.cast(prob_color_augment, tf.float32)),
        true_fn=lambda: color_augment(frames),
        false_fn=lambda: frames)

    # Eventually applies color drop.
    coin_toss_color_drop = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.float32)
    frames = tf.cond(
        pred=tf.less(coin_toss_color_drop, tf.cast(prob_color_drop, tf.float32)),
        true_fn=lambda: color_drop(frames),
        false_fn=lambda: frames)
    result = {**inputs}
    result['video'] = frames

    return result


def add_default_data_augmentation(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        default_color_augmentation_fn, num_parallel_calls=tf.data.AUTOTUNE)


def create_point_tracking_dataset(
    data_dir=None,
    color_augmentation=True,
    train_size=(256, 256),
    shuffle_buffer_size=256,
    split='train',
    # batch_dims=tuple(),
    batch_size=1,
    repeat=True,
    vflip=False,
    random_crop=True,
    tracks_to_sample=256,
    sampling_stride=4,
    max_seg_id=40,
    max_sampled_frac=0.1,
    num_parallel_point_extraction_calls=16,
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
            max_sampled_frac=max_sampled_frac),
        num_parallel_calls=num_parallel_point_extraction_calls)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)

    ds = ds.batch(batch_size)

    if color_augmentation:
        ds = add_default_data_augmentation(ds)
    ds = tfds.as_numpy(ds)

    it = iter(ds)
    while True:
        data = next(it)
        yield data


class KubricData:
    def __init__(
            self, 
            global_rank,
            world_size,
            data_dir,
            **kwargs
        ):
        self.global_rank = global_rank
        self.world_size = world_size

        if self.global_rank == 0:
            self.data = create_point_tracking_dataset(
                data_dir=data_dir,
                **kwargs
            )
      
    def __getitem__(self, idx):
        if self.global_rank == 0:
            batch_all = next(self.data)
            batch_list = []

            batch_size = batch_all['video'].shape[0] // self.world_size


            for i in range(self.world_size):
                batch = {}
                for k, v in batch_all.items():
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        batch[k] = torch.tensor(v[i * batch_size: (i + 1) * batch_size])
                batch_list.append(batch)
        else:
            batch_list = [None] * self.world_size

        
        batch = [None]
        torch.distributed.scatter_object_list(batch, batch_list, src=0)
        
        return batch[0]


if __name__ == '__main__':
    
    import torch.nn as nn
    import lightning as L
    from lightning.pytorch.strategies import DDPStrategy

    class Model(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(256 * 256 * 3 * 24, 1)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            breakpoint()
            x = batch['video']
            x = x.reshape(x.shape[0], -1)
            y = self(x)
            return y

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    model = Model()

    trainer = L.Trainer(accelerator="cpu", strategy=DDPStrategy(), max_steps=1000, devices=1)

    dataloader = KubricData(
        global_rank=trainer.global_rank, 
        data_dir='/media/data2/PointTracking/tensorflow_datasets', 
        batch_size=1 * trainer.world_size,
    )

    trainer.fit(model, dataloader)