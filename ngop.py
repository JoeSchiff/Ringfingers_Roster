
# Desc: Decode a video, set a new gop size, encode the video, then retrieve the new gop size.

# todo
# might be first number (M) -
# check for other missing gets -
# doc globals autodect
# https://imageio.readthedocs.io/en/stable/development/index.html
# container.add_stream("h264", 30) where get name of codecs?
# def max_b_frame_run_in_file(file): should be in class

# GOP size and kfi are not same?
# https://video.stackexchange.com/questions/24680/what-is-keyint-and-min-keyint-and-no-scenecut

# check keyframes using no_key and then get timestamp +
#   and ffprobe
# tested codecs: h264, vp9, mpeg4







import av
import os
from av.datasets import fate as fate_suite




#IN_PATH = fate_suite("h264/bbc2.sample.h264")
#IN_PATH = av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
IN_PATH = fate_suite("h264/interlaced_crop.mp4")

OUT_PATH = os.path.dirname(os.path.realpath(__file__)) + "\keyframe_test.mp4"

NEW_GOP_SIZE = 20



def inspect_keyframes(keyframe_l, msg):
    print(f"\n{msg} keyframe indices: {keyframe_l}")  # Manually inspect keyframe interval
    unique_differnces = set([j-i for i, j in zip(keyframe_l[:-1], keyframe_l[1:])])  # Auto inspect keyframe interval
    print(f"{msg} GOP sizes: {unique_differnces}")  


def get_video_stream(path):
    container = av.open(path)
    stream = container.streams.video[0]
    return container, stream


def get_keyframes(frame, keyframe_l):
    if frame.key_frame:
        keyframe_l.append(frame.index)
    return keyframe_l


def create_video():
    original_keyframes = []
    in_container, in_stream = get_video_stream(IN_PATH)
    
    out_container = av.open(OUT_PATH, "w")
    out_stream = out_container.add_stream("h264", 30)
    out_stream.codec_context.gop_size = NEW_GOP_SIZE
    
    for frame in in_container.decode(in_stream):
        original_keyframes = get_keyframes(frame, original_keyframes)
        packet = out_stream.encode(frame)
        out_container.mux(packet)

    out_stream.encode(None)
    in_container.close()
    out_container.close()
    return original_keyframes


def check_new_video():
    new_keyframes = []
    
    new_container, new_stream = get_video_stream(OUT_PATH)
    for frame in new_container.decode(new_stream):
        new_keyframes = get_keyframes(frame, new_keyframes)

    new_container.close()
    return new_keyframes


original_keyframes = create_video()
new_keyframes = check_new_video()

inspect_keyframes(original_keyframes, "Original")
inspect_keyframes(new_keyframes, "New")









"""
`VideoCodecContext.gop_size.__set__()` will only create new keyframes. It will not replace existing keyframes.

Here is an example of a video with a GOP size of 24. The keyframe indices are:
`[0, 24, 48, 72, 96, 120]`

After setting `gop_size` to 20 and encoding:
`[0, 20, 24, 44, 48, 68, 72, 92, 96, 116, 120]`

This means that setting `gop_size` will have no effect if it is higher than the original value.

Possible solutions:
We can change the name of `gop_size` to `max_gop_size` or change the documentation to explain the situtation. Or both.
expose keyint_min? 






Exploring the ffmpeg documentation at https://ffmpeg.org/ffmpeg-codecs.html:
GOP size is determined by 2 options:
`-g` sets the maximum distance between key frames. 
`keyint_min` sets the minimum distance between key frames.
 

Write test to assert all (or lowest) gop sizes are >= gop_size
all(isinstance(i, int) for i in l)
all(i > gop_size for i in l)

location?
tests/test_codec_context.py
tests/test_encode.py
encode_file_with_max_b_frames




Sets the maximum number of frames between keyframes. Used only for encoding.

:type: int

"""











