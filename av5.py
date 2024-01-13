
"""
how to increase perf
    1: hw accel decoder: unlikely
    2: keyframe jumping
if keyframe interval is less than 1.1166667 sec, then decoding every keyframe should be faster.
decoding keyframes takes longer than decoding non-keyframes.
dont decode every keyframe. decode one keyframe per 1.1166667 sec.
either seek or skip_frame

    how determine the keyframe interval
decode many continuous frames. check if interval is consistant
use NONKEY. check timestamp


    3 modes (fastest to slowest):
1. seek to keyframe (if keyframe interval less than 0.55833335 sec ???)
2. decode all keyframes (if keyframe interval less than 1.1166667 sec)
3. seek to timestamp/frame (if keyframe interval greater than 1.1166667 sec)
"""









import time
import pyav as av
import av.datasets





path = r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4"
container = av.open(path)

total_start_time = time.time()
frame_time_l = []
keyframe_l = []

container.streams.video[0].thread_type = "AUTO"
container.streams.video[0].codec_context.skip_frame = "NONKEY"

#print(container.streams[0].guessed_rate)

prev = 0
for packet in container.demux():
    if packet.stream.type != "video": continue
    frame_start_time = time.time()
    #print('bbbb')
    for frame in packet.decode():
        frame.index
        ft = time.time() - frame_start_time
        frame_time_l.append(ft)
        if len(keyframe_l) < 6:
            if frame.key_frame:
                keyframe_l.append(round(frame.time - prev, 4))
                prev = frame.time
        elif len(keyframe_l) == 6:
            print(keyframe_l[1:-1] == keyframe_l[2:])
        frame_index = round(frame.time * float(s.guessed_rate))
        print(frame, frame.key_frame, frame_index)
        if frame.pts != frame.dts:print(111111111111)
        frame_start_time = time.time()

frame_time_avg = sum(frame_time_l) / len(frame_time_l)
total_end_time = time.time() - total_start_time

print(container.streams.video[0].thread_type, container.streams.video[0].codec_context.skip_frame)
print("Avg {:.4f}s.".format(frame_time_avg))
print("Total {:.4f}s.".format(total_end_time))








path = r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4"
container = av.open(path)


frame_num = 200 # the frame I want
framerate = container.streams.video[0].average_rate # get the frame rate
time_base = container.streams.video[0].time_base # get the time base
sec = int(frame_num/framerate) # timestamp for that frame_num
container.seek(sec*1000000, backward=True, any_frame=False)  # seek to that nearest timestamp
frame = next(container.decode(video=0)) # get the next available frame
sec_frame = int(frame.pts * time_base * framerate) # get the proper key frame number of that timestamp
print(f"{sec_frame=}")

for _ in range(sec_frame, frame_num):
    frame = next(container.decode(video=0))
    print(frame, frame.key_frame)








path = r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4"
container = av.open(path)

total_start_time = time.time()
frame_time_l = []

stream = container.streams.video[0]
stream.thread_type = "AUTO"
#container.streams.video[0].codec_context.skip_frame = "NONKEY"

duration = stream.duration
tb = stream.time_base.denominator
seek_amt = int(tb * 1.5)
seek_amt = 1001

pts = 1001
container.seek(pts, backward=False, any_frame=True, stream=stream)

while pts < duration:
    container.seek(pts, backward=False, any_frame=True, stream=stream)
    frame_start_time = time.time()
    frame = next(container.decode(video=0))
    frame_time_l.append(time.time() - frame_start_time)
    print(frame, frame.key_frame)
    pts += seek_amt

frame_time_avg = sum(frame_time_l) / len(frame_time_l)
total_end_time = time.time() - total_start_time

print(stream.thread_type, stream.codec_context.skip_frame)
print("Avg {:.4f}s.".format(frame_time_avg))
print("Total {:.4f}s.".format(total_end_time))














