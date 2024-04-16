
from imageio import get_reader
import time



video_path = r"C:\Users\jschiffler\Desktop\rl\rl_short.mp4"




reader = get_reader(video_path)

frame_read_l = []

vid_frame_total = reader.count_frames()
vid_frame_count = 0
frame_count_interval = 120


while vid_frame_count + frame_count_interval <= vid_frame_total:
    try:
        time_vcap = time.perf_counter()
        vid_frame_count += frame_count_interval

        #reader._skip_frames(frame_count_interval)
        #frame_na = reader._read_frame()[0]  ## or frame_na = reader.get_data(vid_frame_count)
        frame_na = reader.get_data(vid_frame_count)

        frame_read_l.append((time.perf_counter() - time_vcap))
    except:
        break




print('frame read ave:', sum(frame_read_l) / len(frame_read_l), len(frame_read_l))









