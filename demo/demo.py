import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "locotrack_pytorch"))
import uuid

import gradio as gr
import mediapy
import numpy as np
import cv2
import matplotlib
import torch

from models.locotrack_model import load_model
from viz_utils import paint_point_track


PREVIEW_WIDTH = 768 # Width of the preview video
VIDEO_INPUT_RESO = (256, 256) # Resolution of the input video
POINT_SIZE = 4 # Size of the query point in the preview video
FRAME_LIMIT = 300 # Limit the number of frames to process


def get_point(frame_num, video_queried_preview, query_points, query_points_color, query_count, evt: gr.SelectData):
    print(f"You selected {(evt.index[0], evt.index[1], frame_num)}")

    current_frame = video_queried_preview[int(frame_num)]

    # Get the mouse click
    query_points[int(frame_num)].append((evt.index[0], evt.index[1], frame_num))

    # Choose the color for the point from matplotlib colormap
    color = matplotlib.colormaps.get_cmap("gist_rainbow")(query_count % 20 / 20)
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    print(f"Color: {color}")
    query_points_color[int(frame_num)].append(color)

    # Draw the point on the frame
    x, y = evt.index
    current_frame_draw = cv2.circle(current_frame, (x, y), POINT_SIZE, color, -1)

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw

    # Update the query count
    query_count += 1
    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count # Updated query count
    )


def undo_point(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    if len(query_points[int(frame_num)]) == 0:
        return (
            video_queried_preview[int(frame_num)],
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        )

    # Get the last point
    query_points[int(frame_num)].pop(-1)
    query_points_color[int(frame_num)].pop(-1)

    # Redraw the frame
    current_frame_draw = video_preview[int(frame_num)].copy()
    for point, color in zip(query_points[int(frame_num)], query_points_color[int(frame_num)]):
        x, y, _ = point
        current_frame_draw = cv2.circle(current_frame_draw, (x, y), POINT_SIZE, color, -1)

    # Update the query count
    query_count -= 1

    # Update the frame
    video_queried_preview[int(frame_num)] = current_frame_draw
    return (
        current_frame_draw, # Updated frame for preview
        video_queried_preview, # Updated preview video
        query_points, # Updated query points
        query_points_color, # Updated query points color
        query_count # Updated query count
    )


def clear_frame_fn(frame_num, video_preview, video_queried_preview, query_points, query_points_color, query_count):
    query_count -= len(query_points[int(frame_num)])

    query_points[int(frame_num)] = []
    query_points_color[int(frame_num)] = []

    video_queried_preview[int(frame_num)] = video_preview[int(frame_num)].copy()

    return (
        video_preview[int(frame_num)], # Set the preview frame to the original frame
        video_queried_preview, 
        query_points, # Cleared query points
        query_points_color, # Cleared query points color
        query_count # New query count
    )



def clear_all_fn(frame_num, video_preview):
    return (
        video_preview[int(frame_num)],
        video_preview.copy(),
        [[] for _ in range(len(video_preview))],
        [[] for _ in range(len(video_preview))],
        0
    )


def choose_frame(frame_num, video_preview_array):
    return video_preview_array[int(frame_num)]


def extract_feature(video_input, model_size="small"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    model = load_model(model_size=model_size).to(device)

    video_input = (video_input / 255.0) * 2 - 1
    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)

    with torch.autocast(device_type=device, dtype=dtype):
        with torch.no_grad():
            feature = model.get_feature_grids(video_input)
    
    return feature


def preprocess_video_input(video_path, model_size):
    video_arr = mediapy.read_video(video_path)
    video_fps = video_arr.metadata.fps
    num_frames = video_arr.shape[0]
    if num_frames > FRAME_LIMIT:
        gr.Warning(f"The video is too long. Only the first {FRAME_LIMIT} frames will be used.", duration=5)
        video_arr = video_arr[:FRAME_LIMIT]
        num_frames = FRAME_LIMIT

    # Resize to preview size for faster processing, width = PREVIEW_WIDTH
    height, width = video_arr.shape[1:3]
    new_height, new_width = int(PREVIEW_WIDTH * height / width), PREVIEW_WIDTH

    preview_video = mediapy.resize_video(video_arr, (new_height, new_width))
    input_video = mediapy.resize_video(video_arr, VIDEO_INPUT_RESO)

    preview_video = np.array(preview_video)
    input_video = np.array(input_video)

    video_feature = extract_feature(input_video, model_size)
    
    return (
        video_arr, # Original video
        preview_video, # Original preview video, resized for faster processing
        preview_video.copy(), # Copy of preview video for visualization
        input_video, # Resized video input for model
        video_feature, # Extracted feature
        video_fps, # Set the video FPS
        gr.update(open=False), # Close the video input drawer
        model_size, # Set the model size
        preview_video[0], # Set the preview frame to the first frame
        gr.update(minimum=0, maximum=num_frames - 1, value=0, interactive=True), # Set slider interactive
        [[] for _ in range(num_frames)], # Set query_points to empty
        [[] for _ in range(num_frames)], # Set query_points_color to empty
        [[] for _ in range(num_frames)], 
        0, # Set query count to 0
        gr.update(interactive=True), # Make the buttons interactive
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def track(
    model_size, 
    video_preview,
    video_input, 
    video_feature, 
    video_fps, 
    query_points, 
    query_points_color, 
    query_count, 
):
    if query_count == 0:
        gr.Warning("Please add query points before tracking.", duration=5)
        return None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    # Convert query points to tensor, normalize to input resolution
    query_points_tensor = []
    for frame_points in query_points:
        query_points_tensor.extend(frame_points)
    
    query_points_tensor = torch.tensor(query_points_tensor).float()
    query_points_tensor *= torch.tensor([
        VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0], 1
    ]) / torch.tensor([
        [video_preview.shape[2], video_preview.shape[1], 1]
    ])
    query_points_tensor = query_points_tensor[None].flip(-1).to(device, dtype) # xyt -> tyx

    # Preprocess video input
    video_input = (video_input / 255.0) * 2 - 1
    video_input = torch.tensor(video_input).unsqueeze(0).to(device, dtype)

    model = load_model(model_size=model_size).to(device)
    with torch.autocast(device_type=device, dtype=dtype):
        with torch.no_grad():
            output = model(video_input, query_points_tensor, feature_grids=video_feature)

    tracks = output['tracks'][0].cpu()
    tracks = tracks * torch.tensor([
        video_preview.shape[2], video_preview.shape[1]
    ]) / torch.tensor([
        VIDEO_INPUT_RESO[1], VIDEO_INPUT_RESO[0]
    ])
    tracks = tracks.numpy()


    occlusion_logits = output['occlusion']
    pred_occ = torch.sigmoid(occlusion_logits)
    if 'expected_dist' in output:
        expected_dist = output['expected_dist']
        pred_occ = 1 - (1 - pred_occ) * (1 - torch.sigmoid(expected_dist))

    pred_occ = (pred_occ > 0.5)[0].cpu().numpy()

    # make color array
    colors = []
    for frame_colors in query_points_color:
        colors.extend(frame_colors)
    colors = np.array(colors)
    
    painted_video = paint_point_track(
        video_preview,
        tracks,
        ~pred_occ,
        colors,
    )

    # save video
    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)

    mediapy.write_video(video_file_path, painted_video, fps=video_fps)

    return video_file_path


with gr.Blocks() as demo:
    video = gr.State()
    video_queried_preview = gr.State()
    video_preview = gr.State()
    video_input = gr.State()
    video_feautre = gr.State()
    video_fps = gr.State(24)
    model_size = gr.State("small")

    query_points = gr.State([])
    query_points_color = gr.State([])
    is_tracked_query = gr.State([])
    query_count = gr.State(0)

    gr.Markdown("# LocoTrack Demo")
    gr.Markdown("This is an interactive demo for LocoTrack. For more details, please refer to the [GitHub repository](https://github.com/KU-CVLAB/LocoTrack) or the [paper](https://arxiv.org/abs/2407.15420).")

    gr.Markdown("## First step: Choose the model size, upload your video or select an example video, and click submit.")
    with gr.Row():
        with gr.Accordion("Your video input", open=True) as video_in_drawer:
            model_size_selection = gr.Radio(
                label="Model Size",
                choices=["small", "base"],
                value="small",
            )
            video_in = gr.Video(label="Video Input", format="mp4")
            submit = gr.Button("Submit", scale=0)
    
    gr.Markdown("## Second step: Add query points to the video, and click track.")
    with gr.Row():

        with gr.Column():
            with gr.Row():
                query_frames = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1, label="Choose Frame", interactive=False)
            with gr.Row():
                undo = gr.Button("Undo", interactive=False)
                clear_frame = gr.Button("Clear Frame", interactive=False)
                clear_all = gr.Button("Clear All", interactive=False)

            with gr.Row():
                current_frame = gr.Image(
                    label="Click to add query points", 
                    type="numpy",
                    interactive=False
                )
            
            with gr.Row():
                track_button = gr.Button("Track", interactive=False)

        with gr.Column():
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
                autoplay=True,
                loop=True,
            )
    
    submit.click(
        fn = preprocess_video_input, 
        inputs = [video_in, model_size_selection], 
        outputs = [
            video,
            video_preview,
            video_queried_preview,
            video_input,
            video_feautre,
            video_fps,
            video_in_drawer,
            model_size,
            current_frame,
            query_frames,
            query_points,
            query_points_color,
            is_tracked_query,
            query_count,
            undo,
            clear_frame,
            clear_all,
            track_button,
        ],
        queue = False
    )

    query_frames.change(
        fn = choose_frame,
        inputs = [query_frames, video_queried_preview],
        outputs = [
            current_frame,
        ],
        queue = False
    )

    current_frame.select(
        fn = get_point, 
        inputs = [
            query_frames,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count,
        ], 
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ], 
        queue = False
    )
    
    undo.click(
        fn = undo_point,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_frame.click(
        fn = clear_frame_fn,
        inputs = [
            query_frames,
            video_preview,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    clear_all.click(
        fn = clear_all_fn,
        inputs = [
            query_frames,
            video_preview,
        ],
        outputs = [
            current_frame,
            video_queried_preview,
            query_points,
            query_points_color,
            query_count
        ],
        queue = False
    )

    track_button.click(
        fn = track,
        inputs = [
            model_size,
            video_preview,
            video_input,
            video_feautre,
            video_fps,
            query_points,
            query_points_color,
            query_count,
        ],
        outputs = [
            output_video,
        ],
        queue = True,
    )

demo.launch(show_api=False, show_error=True, debug=True)