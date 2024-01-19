
# Desc: Test frame indices

"""
findings:
repeating incorrect frame index
always at end of video
idx is 1 higher than total frames
NONE and SLICE are same. FRAME and AUTO are same
SLICE is usually off by 1
FRAME number of repeat frames is related to thread count
sometimes stream.frames also wrong


separate test methods or reuse infrastructure?
General question about writing new tests: Would you prefer if I made separate test 
Some of the methods are getting large.
If you're not concerned about the tests running slightly longer, then it could help keep things organized.

what does this do? if isinstance(frame, Frame):
remove frame.index = self.ptr.frame_number - 1?

use frame_number - correction instead of index?


All frames in the list returned by '_send_packet_and_recv' will have the same index. This will reassign the correct indices.


Solution for [Issue 1158](https://github.com/PyAV-Org/PyAV/issues/1158).

`stream.thread_type.FRAME` threading causes an incorrect series of repeating frame indices at the end of most videos. This is because a list of frames are returned by `CodecContext._send_packet_and_recv` https://github.com/PyAV-Org/PyAV/blob/1962443425d9f63c80d220546ac5f3dfb0799f6f/av/codec/context.pyx#L405 . `frame.index` corresponds to the most recent frame received from the decoder. Therefore all frames in the list are assigned the same index as the final frame in the list.

I solved this issue by detecting whenever multiple frames are returned and then correcting the `frame.index` based on the frame's position in the list.

#The test includes only `SLICE` and `AUTO` because `NONE` and `SLICE` demonstrate the same index behavior. `FRAME` and `AUTO` demonstrate the same index behavior.
AUTO covers all threading options?

`AVCodecContext::frame_number` is deprecated. I switched to `AVCodecContext::frame_num`.
https://ffmpeg.org//doxygen/trunk/structAVCodecContext.html#a1606ae31b14af5a2940b94c99264c0fc
include/libavcodec/avcodec.pxd
av/codec/context.pyx
"""




import av
from av.datasets import fate as fate_suite
import os




PATHS = (fate_suite("h264/bbc2.sample.h264"), av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4"), fate_suite("h264/interlaced_crop.mp4"))
THREAD_TYPES = ("SLICE", "AUTO")
THREAD_COUNTS = 5



def compare_indices(path, thread_type, thread_count):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = thread_type
    stream.thread_count = thread_count

    frame_list = []
    compare_list = []
    for idx, frame in enumerate(container.decode(stream)):
        frame_list.append(frame.index)
        compare_list.append(idx)
    if frame_list != compare_list:
        print(f'{os.path.basename(path)} {thread_type} {thread_count}')
        print(frame_list[-2], compare_list[-2])
    
    container.close()


for path in PATHS:
    for thread_type in THREAD_TYPES:
        for thread_count in range(THREAD_COUNTS):
            compare_indices(path, thread_type, thread_count)























