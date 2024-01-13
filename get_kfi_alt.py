
# Desc: Check kfi with nokey, get ts, convert to frame index




import av
from av.datasets import fate as fate_suite




#IN_PATH = fate_suite("h264/bbc2.sample.h264")
#IN_PATH = av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
IN_PATH = fate_suite("h264/interlaced_crop.mp4")


keyframe_l = []
def timestamp_to_frame_index(timestamp, stream):
    fps = stream.average_rate
    print(stream.average_rate, stream.guessed_rate, stream.codec_context.ticks_per_frame)
    time_base = stream.time_base
    start_time = stream.start_time
    frame_index = round((timestamp - start_time) * float(time_base) * float(fps))
    keyframe_l.append(frame_index)


container = av.open(IN_PATH)


stream = container.streams.video[0]
stream.thread_type = "AUTO"
stream.codec_context.skip_frame = "NONKEY"


for packet in container.demux():
    if packet.stream.type != "video": continue
    for frame in packet.decode():
        if frame.pts != frame.dts: print(f"{frame.pts=} {frame.dts=}")
        timestamp_to_frame_index(frame.pts, stream)

print(keyframe_l)

















