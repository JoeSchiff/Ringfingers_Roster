

import av
from av import VideoFrame
from fractions import Fraction



PATH = r"C:\Users\jschiffler\Desktop\pyav\tnew.mp4"

WIDTH = 320
HEIGHT = 240
DURATION = 100


with av.open(PATH, "w") as output:
    stream = output.add_stream("mpeg4", 24)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"
    for i in range(DURATION):
        frame = VideoFrame(WIDTH, HEIGHT, "rgb24")
        frame.pts = i * 2000
        frame.time_base = Fraction(1, 48000)
        for packet in stream.encode(frame):
            packet.is_corrupt = True
            output.mux(packet)



with av.open(PATH, "r") as container:
    stream = container.streams.video[0]
    for packet in container.demux(stream):
        if packet.is_corrupt:
            print(f"\n{packet.pts=}")























