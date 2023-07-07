

import time
import av
import av.datasets



print("Decoding with default (slice) threading...")

container = av.open(
    av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
)

start_time = time.perf_counter()
for packet in container.demux():
    #print(packet)
    for frame in packet.decode():
        #print(frame)
        pass

default_time = time.perf_counter() - start_time
container.close()


print("Decoding with auto threading...")

container = av.open(
    av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")
)

# !!! This is the only difference.
container.streams.video[0].thread_type = "AUTO"

start_time = time.perf_counter()
for packet in container.demux():
    #print(packet)
    for frame in packet.decode():
        #print(frame)
        pass

auto_time = time.perf_counter() - start_time
container.close()


print("Decoded with default threading in {:.2f}s.".format(default_time))
print("Decoded with auto threading in {:.2f}s.".format(auto_time))





flags = container.format.flags.name
bit_rate = container.bit_rate

video_stream = container.streams.video[0]

frame = video_stream.decode

average_rate = video_stream.average_rate
base_rate
 

frames
duration





import time
import av
import av.datasets




path = r"C:\Users\jschiffler\Desktop\rr\short_test.mp4"



with av.open(path) as container:
    stream = container.streams.video[0]
    #stream.codec_context.skip_frame = "NONKEY"  # keyframes only
    start_time = time.perf_counter()   
    for frame in container.decode(stream):
        na = frame.to_ndarray
        print(time.perf_counter() - start_time, frame)
        start_time = time.perf_counter()
        #container.seek(ts)



        
container = av.open(path)
stream = container.streams.video[0]
#stream.codec_context.skip_frame = "NONKEY"  # keyframes only
start_time = time.perf_counter()
gen = container.decode(stream)
frame = next(gen)
frame.time


container.seek(33350)
gen = container.decode(stream)
frame = next(gen)
frame.time
frame.pts





        na = frame.to_ndarray
        print(time.perf_counter() - start_time, frame)
        start_time = time.perf_counter()
        #container.seek(ts)



fps = 30

fci = fps*1.1166667

seek_value = stream.time_base.denominator*fci


print(format(float(stream.time_base), 'f'))










