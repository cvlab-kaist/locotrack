from moviepy.editor import VideoFileClip, clips_array

# Load your video files
video1 = VideoFileClip("videos/swing-CoTracker.mp4")
video2 = VideoFileClip("videos/swing-TAPIR.mp4")
video3 = VideoFileClip("videos/swing-Ours.mp4")

# Make sure all videos have the same height
max_height = max(video1.h, video2.h, video3.h)

video1 = video1.resize(height=max_height)
video2 = video2.resize(height=max_height)
video3 = video3.resize(height=max_height)

# Concatenate the videos horizontally
final_video = clips_array([[video1, video2, video3]])

# Write the output to a file
final_video.write_videofile("videos/swing-merge.mp4")
