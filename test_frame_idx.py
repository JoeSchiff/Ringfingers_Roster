
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

ideas:
https://github.com/PyAV-Org/PyAV/blob/1962443425d9f63c80d220546ac5f3dfb0799f6f/av/frame.pyx#L33
maybe flushing frames?
loop through thread counts +
test older pyav versions +
recreate in ffmpeg -
what does AUTO mean?
av/codec/codec.pyx: "AUTO_THREADS", lib.AV_CODEC_CAP_OTHER_THREADS
av/codec/context.pyx: "AUTO", lib.FF_THREAD_SLICE | lib.FF_THREAD_FRAME
ptr.thread_type = 2
try updating index with ptr.frame_number before every print
"""



import av
from av.datasets import fate as fate_suite
import os
import json




#PATHS = (fate_suite("h264/bbc2.sample.h264"), av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4"), fate_suite("h264/interlaced_crop.mp4"))
PATHS = (r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4", fate_suite("h264/bbc2.sample.h264"), av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4"), fate_suite("h264/interlaced_crop.mp4"))
THREAD_TYPES = ("NONE", "SLICE", "FRAME", "AUTO")
THREAD_COUNTS = 7



def compare_indices(path, short_path, thread_type, thread_count):
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = thread_type
    stream.thread_count = thread_count
    print(f'{short_path} {thread_type} {thread_count}')

    err_count = 0
    for idx, frame in enumerate(container.decode(stream)):
        if frame.index != idx:
            #print(idx, frame.index)
            err_count += 1
    results_d[short_path][thread_type][thread_count] = err_count
    
    container.close()


results_d = {}
for path in PATHS:
    short_path = os.path.basename(path)
    results_d[short_path] = {}
    
    for thread_type in THREAD_TYPES:
        results_d[short_path][thread_type] = {}
        
        for thread_count in range(THREAD_COUNTS):
            results_d[short_path][thread_type][thread_count] = []
            compare_indices(path, short_path, thread_type, thread_count)




print('\n Key is thread_count. Value is number of errors.')
print(json.dumps(results_d, indent=4))



















