
# Desc: Find all ffmpeg struct names in pyav

"""
todo:
iter pyav structs +
    dynamic or static?
iter pyav data fields +
compare with ffmpeg data fields
check for deprecation
iter ffmpeg versions

instead: iter all deprecated data fields?

list which file each struct is found in?


"""



from glob import glob

pyav_path = r"C:\Users\jschiffler\Desktop\pyav\PyAV-main\include"

#structs = ['AVAudioFifo', 'AVClass', 'AVCodec', 'AVCodecContext', 'AVCodecDescriptor', 'AVCodecParameters', 'AVCodecParser', 'AVCodecParserContext', 'AVComponentDescriptor', 'AVFilter', 'AVFilterContext', 'AVFilterGraph', 'AVFilterInOut', 'AVFilterLink', 'AVFilterPad', 'AVFormatContext', 'AVFrame', 'AVFrameSideData', 'AVIOContext', 'AVIOInterruptCB', 'AVInputFormat', 'AVMotionVector', 'AVOption', 'AVOption_default_val', 'AVOutputFormat', 'AVPacket', 'AVPacketSideData', 'AVPixFmtDescriptor', 'AVProbeData', 'AVStream', 'AVSubtitle', 'AVSubtitleRect', 'SwrContext', 'SwsContext', 'SwsFilter']


struct_d = {}

def get_indent_amount(line):
    return len(line) - len(line.lstrip(" "))

struct_indent_amount = False
for file_path in glob(pyav_path + "\**\*.pxd", recursive=True):
    print(file_path)
    f = open(file_path, "r")
    for line in f.readlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if line.strip().startswith("cdef struct"):
            struct = line.split("cdef struct ")[-1].split(':\n')[0].split()[0]
            struct_indent_amount = get_indent_amount(line)
            print(11111, struct, struct_indent_amount)
            struct_d[struct] = []
        elif struct_indent_amount:
            line_indent_amount = get_indent_amount(line)
            #print(struct_indent_amount, line_indent_amount, line)
            if line_indent_amount == struct_indent_amount + 4:
                struct_d[struct].append(line)
            else:
                print('new line')
                struct_indent_amount = False


for k,v in struct_d.items():
    print('\n', k)
    for i in v:
        print(i.split()[-1].strip())
            



import requests


#url = "https://ffmpeg.org/doxygen/trunk/classes.html"
#resp = requests.get(url)

for struct, v in struct_d.items():
    url = f"https://ffmpeg.org/doxygen/trunk/struct{struct}.html"
    resp = requests.get(url)

    if i in resp.text:
        print(i)
    else:
        print(11111, i)






"""
Desc: get var names from inside python env
This doesn't work because it doesn't tell you the struct.
Also, frame.index / frame_num is not detected.
Prob need to init a video and use decode function.


import av

nd = {}
skip_list = ["os", "sys", "_core"]

def get_dict(obj):
    if obj.startswith("__") or obj in skip_list:
        print('skip:', obj)
        return
    try:
        for k,v in obj.items():
            get_dict(v)
    except Exception as err:
        #print(55555, err, obj)
        return obj

for obj in av.__dict__:
    final_obj = get_dict(obj)
    if final_obj:
        print('\n', final_obj)
        nd[final_obj] = []
        try:
            for i in av.__dict__[final_obj].__dict__:
                if i.startswith("__"):
                    continue
                nd[final_obj].append(i)
        except:
            print(final_obj)
    

for k,v in nd.items():
    if v:
        print('\n', k)
        for i in v:
            print(i)
"""




