

import time
import cv2


video_path = r"C:\Users\jschiffler\Desktop\rl\rl_short.mp4"

frame_read_l = []
vid_frame_count = 0

frame_count_interval = 120


vcap = cv2.VideoCapture(video_path)
vid_frame_total = vcap.get(7)



while vid_frame_count + frame_count_interval <= vid_frame_total:

    time_frame_read = time.perf_counter()
    vid_frame_count += frame_count_interval
    
    vcap.set(1, vid_frame_count) 
    frame_na = vcap.read()
    
    frame_read_l.append((time.perf_counter() - time_frame_read))





print('frame read ave:', sum(frame_read_l) / len(frame_read_l), len(frame_read_l))









