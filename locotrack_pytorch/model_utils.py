from typing import Sequence, Optional

import torch
import torch.nn.functional as F

from models.utils import convert_grid_coordinates
from data.evaluation_datasets import compute_tapvid_metrics

def huber_loss(tracks, target_points, occluded, delta=4.0, reduction_axes=(1, 2)):
    """Huber loss for point trajectories."""
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)  # add eps to prevent nan
    loss_huber = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))
    loss_huber = loss_huber * (1.0 - occluded.float())

    if reduction_axes:
        loss_huber = torch.mean(loss_huber, dim=reduction_axes)

    return loss_huber

def prob_loss(tracks, expd, target_points, occluded, expected_dist_thresh=8.0, reduction_axes=(1, 2)):
    """Loss for classifying if a point is within pixel threshold of its target."""
    err = torch.sum((tracks - target_points) ** 2, dim=-1)
    invalid = (err > expected_dist_thresh ** 2).float()
    logprob = F.binary_cross_entropy_with_logits(expd, invalid, reduction='none')
    logprob = logprob * (1.0 - occluded.float())
    
    if reduction_axes:
        logprob = torch.mean(logprob, dim=reduction_axes)
        
    return logprob

def tapnet_loss(points, occlusion, target_points, target_occ, shape, mask=None, expected_dist=None,
                position_loss_weight=0.05, expected_dist_thresh=6.0, huber_loss_delta=4.0, 
                rebalance_factor=None, occlusion_loss_mask=None):
    """TAPNet loss."""
    
    if mask is None:
        mask = torch.tensor(1.0)

    points = convert_grid_coordinates(points, shape[3:1:-1], (256, 256), coordinate_format='xy')
    target_points = convert_grid_coordinates(target_points, shape[3:1:-1], (256, 256), coordinate_format='xy')

    loss_huber = huber_loss(points, target_points, target_occ, delta=huber_loss_delta, reduction_axes=None) * mask
    loss_huber = torch.mean(loss_huber) * position_loss_weight

    if expected_dist is None:
        loss_prob = torch.tensor(0.0)
    else:
        loss_prob = prob_loss(points.detach(), expected_dist, target_points, target_occ, expected_dist_thresh, reduction_axes=None) * mask
        loss_prob = torch.mean(loss_prob)

    target_occ = target_occ.to(dtype=occlusion.dtype)
    loss_occ = F.binary_cross_entropy_with_logits(occlusion, target_occ, reduction='none') * mask

    if rebalance_factor is not None:
        loss_occ = loss_occ * ((1 + rebalance_factor) - rebalance_factor * target_occ)
        
    if occlusion_loss_mask is not None:
        loss_occ = loss_occ * occlusion_loss_mask

    loss_occ = torch.mean(loss_occ)

    return loss_huber, loss_occ, loss_prob


def tapir_loss(
    batch, 
    output,
    position_loss_weight=0.05,
    expected_dist_thresh=6.0,
):
    loss_scalars = {}
    loss_huber, loss_occ, loss_prob = tapnet_loss(
        output['tracks'],
        output['occlusion'],
        batch['target_points'],
        batch['occluded'],
        batch['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
        expected_dist=output['expected_dist']
        if 'expected_dist' in output
        else None,
        position_loss_weight=position_loss_weight,
        expected_dist_thresh=expected_dist_thresh,
    )
    loss = loss_huber + loss_occ + loss_prob
    loss_scalars['position_loss'] = loss_huber
    loss_scalars['occlusion_loss'] = loss_occ
    if 'expected_dist' in output:
        loss_scalars['prob_loss'] = loss_prob

    if 'unrefined_tracks' in output:
        for l in range(len(output['unrefined_tracks'])):
            loss_huber, loss_occ, loss_prob = tapnet_loss(
                output['unrefined_tracks'][l],
                output['unrefined_occlusion'][l],
                batch['target_points'],
                batch['occluded'],
                batch['video'].shape,  # pytype: disable=attribute-error  # numpy-scalars
                expected_dist=output['unrefined_expected_dist'][l]
                if 'unrefined_expected_dist' in output
                else None,
                position_loss_weight=position_loss_weight,
                expected_dist_thresh=expected_dist_thresh,
            )
            loss = loss + loss_huber + loss_occ + loss_prob
            loss_scalars[f'position_loss_{l}'] = loss_huber
            loss_scalars[f'occlusion_loss_{l}'] = loss_occ
            if 'unrefined_expected_dist' in output:
                loss_scalars[f'prob_loss_{l}'] = loss_prob

    loss_scalars['loss'] = loss
    return loss, loss_scalars



def eval_batch(
    batch, 
    output, 
    eval_metrics_resolution = (256, 256),
    query_first = False,
):
    query_points = batch['query_points']
    query_points = convert_grid_coordinates(
        query_points,
        (1,) + batch['video'].shape[2:4],  # (1, height, width)
        (1,) + eval_metrics_resolution,  # (1, height, width)
        coordinate_format='tyx',
    )
    gt_target_points = batch['target_points']
    gt_target_points = convert_grid_coordinates(
        gt_target_points,
        batch['video'].shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )
    gt_occluded = batch['occluded']

    tracks = output['tracks']
    tracks = convert_grid_coordinates(
        tracks,
        batch['video'].shape[3:1:-1],  # (width, height)
        eval_metrics_resolution[::-1],  # (width, height)
        coordinate_format='xy',
    )

    occlusion_logits = output['occlusion']
    pred_occ = torch.sigmoid(occlusion_logits)
    if 'expected_dist' in output:
        expected_dist = output['expected_dist']
        pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))
    pred_occ = pred_occ > 0.5  # threshold

    query_mode = 'first' if query_first else 'strided'
    metrics = compute_tapvid_metrics(
        query_points=query_points.detach().cpu().numpy(),
        gt_occluded=gt_occluded.detach().cpu().numpy(),
        gt_tracks=gt_target_points.detach().cpu().numpy(),
        pred_occluded=pred_occ.detach().cpu().numpy(),
        pred_tracks=tracks.detach().cpu().numpy(),
        query_mode=query_mode,
    )

    return metrics