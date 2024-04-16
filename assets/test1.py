
from imageio import get_reader
import time



video_path = r"C:\Users\jschiffler\Desktop\rl\rl_short.mp4"


reader = get_reader(video_path)

frame_read_l = []


time_vcap = time.perf_counter()

for frame_i, frame_na in enumerate(reader):

    if frame_i % 120 != 0: continue

    frame_read_l.append((time.perf_counter() - time_vcap))
    
    time_vcap = time.perf_counter()



print('frame read ave:', sum(frame_read_l) / len(frame_read_l), len(frame_read_l))







