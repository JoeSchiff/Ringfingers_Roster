
# Desc: Determine the keyframe interval




import time
import av





path = r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4"

total_start_time = time.time()
container = av.open(path)
vcap_time = time.time() - total_start_time

frame_time_l = []
keyframe_l = []

stream = container.streams.video[0]
stream.thread_type = "AUTO"
#stream.codec_context.skip_frame = "NONKEY"

#print(container.streams[0].guessed_rate)


prev = 0
for packet in container.demux():
    if packet.stream.type != "video": continue
    frame_start_time = time.time()
    print('ppp', packet, packet.is_keyframe)
    for frame in packet.decode():
        print(frame, frame.key_frame, frame.index)
        ft = time.time() - frame_start_time
        frame_time_l.append(ft)
        if len(keyframe_l) < 6:
            if frame.key_frame:
                keyframe_l.append(round(frame.time - prev, 4))
                prev = frame.time
        #elif len(keyframe_l) == 6:
            #print(keyframe_l[1:-1] == keyframe_l[2:])
        frame_index = round(frame.time * float(stream.guessed_rate))
        if frame.pts != frame.dts:print(111111111111)
        frame_start_time = time.time()


frame_time_avg = sum(frame_time_l) / len(frame_time_l)
total_end_time = time.time() - total_start_time


print(f"vcap time: {vcap_time:.4f}")
print(stream.thread_type, stream.codec_context.skip_frame)
print("Avg {:.4f}s.".format(frame_time_avg))
print("Total {:.4f}s.".format(total_end_time))












