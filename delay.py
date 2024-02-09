
"""
Desc: Test for packet properties:
AV_PKT_FLAG_KEY: is_keyframe
AV_PKT_FLAG_CORRUPT: is_corrupt
AV_PKT_FLAG_DISCARD: is_discard
AV_PKT_FLAG_TRUSTED: is_trusted
AV_PKT_FLAG_DISPOSABLE: is_disposable



todo:
only get?
create test_packet.py
test default value
assign and test again



https://ffmpeg.org/doxygen/trunk/structAVPacket.html


many different names. is this from ffmpeg?
container: flags, options
codec: capabilities, properties
codec_context: flags, flags2, options
stream: side_data
packet: 'is_corrupt', 'is_keyframe'



packet.is_keyframe does not match frame.key_frame: fate_suite("h264/interlaced_crop.mp4") and fate_suite("h264/bbc2.sample.h264")
always off by 5? -
need to test in ffmpeg
decoder delay?



cc.codec.delay is bool should be int?

https://ffmpeg.org/doxygen/trunk/structAVCodec.html
capabilities ->
AV_CODEC_CAP_DELAY: https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga3f55f5bcfbb12e06c7cb1195028855e6
must flush

delay: https://ffmpeg.org/doxygen/trunk/structAVCodecContext.html#a948993adfdfcd64b81dad1151fe50f33
number of frames

context should have delay
codec should have CAP_DELAY

low delay moved to codec
PyAV/av/codec/context.pyx needs `delay`
"""



import av
from av.datasets import fate as fate_suite



#PATHS = (r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4", fate_suite("h264/bbc2.sample.h264"), av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4"), fate_suite("h264/interlaced_crop.mp4"))
PATH = r"C:\Users\jschiffler\Desktop\text_n_stuff\old_bullshit\2023\rr\short_test.mp4"
PATH = av.datasets.curated("pexels/time-lapse-video-of-night-sky-857195.mp4")


with av.open(PATH) as container:
    stream = container.streams.video[0]
    cc = stream.codec_context
    cc.thread_count = 1
    cc.thread_type = "NONE"
    cc.low_delay = True
    for packet in container.demux(stream):
        print(f"\n{packet.is_keyframe=}")
        for frame in stream.decode(packet):
            if packet.is_keyframe != frame.key_frame:
                print(3333333)
            print(frame.index, frame.key_frame)
    while True:
        for frame in cc.decode(None):
            print(999)































