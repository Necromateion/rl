import numpy as np
from var import *


from collections import deque
import cv2

def process_input_frames(frames):
    max_frame = np.maximum.reduce(frames)
    # Convert to Luminance (grayscale)
    Luminance_frame = rgb_to_luminance(max_frame).astype(np.uint8)
    
    # Resize to 84x84
    resized_frame = cv2.resize(Luminance_frame, dsize=(84, 84))
    return resized_frame

def rgb_to_luminance(frame):
    return 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]


def process_and_stack_frames(new_frame, stacked_frames, is_new_episode):
    processed_frame = process_input_frames([new_frame])

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(AGENT_HISTORY_LENGTH)], maxlen=AGENT_HISTORY_LENGTH)
        # Because we're in a new episode, copy the same frame 4x, create a stack of 4 frames
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
        stacked_frames.append(processed_frame)
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(processed_frame)

    # Stack the frames
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames