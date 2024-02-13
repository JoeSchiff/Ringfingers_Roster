
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
https://ffmpeg.org/doxygen/trunk/group__lavc__packet.html


many different names. is this from ffmpeg?
container: flags, options
codec: capabilities, properties
codec_context: flags, flags2, options
stream: side_data
packet: 'is_corrupt', 'is_keyframe'




output_corrupt
low_delay True is default?
packet.update()

Consider CodecContextFlags, whis is the type 


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
PATH = fate_suite("h264/interlaced_crop.mp4")


pl = []
fl = []

container = av.open(PATH)
stream = container.streams.video[0]
cc = stream.codec_context
cc.thread_count = 1
cc.thread_type = "NONE"
cc.low_delay = True
for packet in container.demux(stream):
    #print('\n', packet)
    #print(f"\n{packet.is_keyframe=}")
    if packet.is_keyframe:
        pl.append(packet.pts)
    for frame in stream.decode(packet):
        if frame.key_frame:
            fl.append(frame.pts)
        #print(frame)
        #if packet.is_keyframe != frame.key_frame:
        #    print(3333333)
        #print(frame.index, frame.key_frame)
while True:
    for frame in cc.decode(None):
        print(999)


print(pl)
print(fl)















"""
#### Overview
`codec.delay` is declared as an integer: https://github.com/PyAV-Org/PyAV/blob/8e32fe7ed7b83eff27682b4e026e7320cb8dfe1e/include/libavcodec/avcodec.pxd#L192
However, trying to access it returns a boolean.

```
import av
from av.datasets import fate as fate_suite

PATH = fate_suite("h264/interlaced_crop.mp4")

container = av.open(PATH)
stream = container.streams.video[0]
print(stream.codec_context.codec.delay)
```
Output:
```
True
```

#### Explanation
PyAV's `codec.delay` is a reference to ffmpeg's [AV_CODEC_CAP_DELAY](https://ffmpeg.org/doxygen/6.0/group__lavc__core.html#ga3f55f5bcfbb12e06c7cb1195028855e6).

The :type: int `delay` is likely supposed to be a reference to ffmpeg's [AVCodecContext::delay](https://ffmpeg.org/doxygen/6.0/structAVCodecContext.html#a948993adfdfcd64b81dad1151fe50f33).

I believe the :type: int `delay` is being overwritten by the :type: bool `delay` here:
https://github.com/PyAV-Org/PyAV/blob/8e32fe7ed7b83eff27682b4e026e7320cb8dfe1e/av/codec/codec.pyx#L313

#### Changes
I created a property `codec_context.delay` which references ffmpeg's `AVCodecContext::delay`.
I renamed `codec.delay` to `codec.cap_delay` to match ffmpeg's `AV_CODEC_CAP_DELAY` and to differentiate from the former.

#### Tests
Since the ffmpeg docs state "Set by libavcodec.", I was unsure how to test these properties. I ended up finding a file in the FATE suite with a detected delay value. I don't know how to verify that the value returned `codec_context.delay` is accurate, so the test passes if any delay is detected.

If this is an acceptable testing strategy, then I'll write up more tests to help merge some of the other PRs.

"""













