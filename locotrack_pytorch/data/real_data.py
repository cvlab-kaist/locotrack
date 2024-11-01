from glob import glob

import cv2
import mediapy
from PIL import Image
from torch.utils.data import Dataset

class RealVideoDataset(Dataset):
    def __init__(self, data_dir, video_size=(384, 512), video_length=24, points_to_sample=256):
        # Recursively search for all mp4 files in the data_dir
        self.data = glob(os.path.join(data_dir, '**', '*.mp4'), recursive=True)
        self.data = sorted(self.data)

        self.video_size = video_size
        self.video_length = video_length
        self.points_to_sample = points_to_sample
        self.strong = RandAugmentMC(n=2, m=10)
        self.sift = cv2.SIFT_create()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = mediapy.read_video(self.data[idx])  # shape (T, H, W, C)
        T, H, W, C = video.shape
        if T != self.video_length:
            # Randomly sample a video segment
            start = np.random.randint(0, T - self.video_length)
            video = video[start: start + self.video_length]
        
        if H != self.video_size[0] or W != self.video_size[1]:
            video = mediapy.resize_video(video, self.video_size)

        # Apply strong augmentation
        strong_video = []
        for frame in video:
            strong_video.append(self.strong(Image.fromarray(frame)))
        strong_video = np.stack(strong_video)

        # Sample salient points uniformly from frames
        sampled_points = []
        for t in range(self.video_length):
            frame = video[t]
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keypoints = self.sift.detect(gray_frame, None)
            points = np.array([[t, kp.pt[1], kp.pt[0]] for kp in keypoints])  # (t, y, x)
            if points.shape[0] > 0:
                sampled_points.extend(points)  # Collect points from all frames

        # rancomly sample points_to_sample points
        if len(sampled_points) > self.points_to_sample:
            sampled_points = np.array(sampled_points)
            sampled_points = sampled_points[np.random.choice(sampled_points.shape[0], self.points_to_sample, replace=False)]
        else:
            # pad with random points
            sampled_points = np.pad(
                sampled_points, ((0, self.points_to_sample - len(sampled_points)), (0, 0)), mode='constant')
        
        return {
            'video': torch.tensor(video).permute(0, 3, 1, 2),
            'strong_video': torch.tensor(strong_video).permute(0, 3, 1, 2),
            'query_points': torch.tensor(sampled_points)
        }
