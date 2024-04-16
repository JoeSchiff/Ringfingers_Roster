
'''
todo:
does perf matter?
skip to end of match?
search only end of video?


concerns:
append to ALL at beginning or end? before: will skip file on fatal error. after: will fail repeatedly
grab sb color regions x4?

'''









from imageio import get_reader
from PIL import Image
import numpy as np
from pytesseract import pytesseract
import time
import json
import sys
import os
import ntpath
import mimetypes






try:
    pytesseract.tesseract_cmd = r"C:\Users\jschiffler\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    pytesseract.get_tesseract_version()
    if 'eng' not in pytesseract.get_languages(): raise
except:
    pytesseract.tesseract_cmd = 'tesseract'
    


inc = 0
def save_img(numpy_array):
    global inc
    inc += 1
    im = Image.fromarray(numpy_array)
    #nameee = '/home/joepers/Desktop/t/here' + str(inc) + '.jpeg'
    nameee = r"C:\Users\jschiffler\Desktop\rl\\" + str(inc) + '.jpeg'
    im.save(nameee)



# Gather all files and options from args
def get_files_f(arg_l, arg_d):

    for item in arg_l:

        # Options
        if item.startswith('--'):
            if item == '--nonrecursive':
                arg_d['recursive'] = False
                print('Option set: nonrecursive')
            elif item == '--noskip':
                arg_d['noskip'] = True
                print('Option set: noskip')
            elif item.startswith('--output='):
                arg_d['output_dir'] = ''.join(item.split('--output=')[1:])
                arg_d['explicit_output'] = True
                print('Option set: output location:', arg_d['output_dir'])
            else:
                print('\nOption not recognized:', item)

        # Directory
        elif os.path.isdir(item):

            # Set output location based on input
            if not arg_d['explicit_output']:  # Set output dir if not explicitly stated
                if not arg_d['output_dir']:  # Output not set
                    arg_d['output_dir'] = item  # Use first dir
                    
                elif not os.path.abspath(arg_d['output_dir']) in os.path.abspath(item):  # Change output from first dir to CWD if not child dir
                    arg_d['output_dir'] = os.getcwd()  # Use CWD as output location when multi dirs are invoked

            # Gather sub dirs only when recursive
            if arg_d['recursive']:
                child_l = [os.path.join(item, child) for child in os.listdir(item)]  # List of items in dir
            else:
                child_l = [os.path.join(item, child) for child in os.listdir(item) if os.path.isfile(os.path.join(item, child))]  # List of files in dir
            get_files_f(child_l, arg_d)  ## does this need to return a new arg_d? arg_d = get_files_f(child_l, arg_d)

        # File
        elif os.path.isfile(item):
            if os.access(item, os.R_OK):
                if ntpath.basename(item).startswith(ntpath.basename(result_filename.rsplit('.')[0])):  # Result file
                    arg_d['result_file_l'].append(item)
                    print('Result file detected:', item)
                else:
                    all_files_l.append(item)  # Video file
            else:
                print('Error: File is not readable:', item)

        else:
            print('Error: Not a file, directory, or option:', item, type(item))


    return arg_d




# Add names to dict
def add_names_f(name, video_path, name_d):
    if name in name_d:  # Name already in dict
        if not video_path in name_d[name]:  # Prevent dups
            print('Adding video to name:', video_path, name)
            name_d[name].append(video_path)
    else:  # New entry as list
        print('Adding video to new name:', video_path, name)
        name_d[name] = [video_path]


# Save results to file
def write_res_f(name_d):
    json_results = json.dumps(name_d, indent=4, ensure_ascii=False)
    with open(output_loc, 'w', errors='replace') as output_file:
        output_file.write(json_results)
    return json_results





print('Arguments:', sys.argv[1:])

# Default input and output directory
default_path = os.path.dirname(os.path.abspath(sys.argv[0]))

# Default options
arg_d = {
'recursive': True,
'noskip': False,
'output_dir': None,
'explicit_output': False,
'result_file_l': []
}

result_filename = "rl_out.txt"


all_files_l = []  # All the videos found
checked_files_l = []  # Used only by get_frames_f to track completed videos 
name_d = {"  ALL  ": []}  # Used by process_frames_f to mark a video as complete and for Resumption
frame_read_l = []



# Get user-supplied option values
arg_d = get_files_f(sys.argv[1:], arg_d)

# If no files as input, default to all files in script/exe dir
if not all_files_l:
    arg_d = get_files_f([default_path], arg_d)

# Set output location if not specified
if not arg_d['output_dir']:
    arg_d['output_dir'] = os.getcwd()

output_loc = os.path.abspath(os.path.join(arg_d['output_dir'], result_filename))





for video_path in all_files_l:

    print('Starting video:', video_path)

    # Check if file is a video
    mimetype_res = mimetypes.guess_type(video_path)[0]
    if not mimetype_res or not mimetype_res.startswith('video'):
        print('File MIME type detected as non-video. Skipping:', video_path, mimetype_res)
        continue
    
    try:
        reader = get_reader(video_path)

    except Exception as errex:
        print('__Error:', errex, sys.exc_info()[2].tb_lineno)
        
    name_d["  ALL  "].append(video_path)

    frame_na = reader._read_frame()[0]  ## fix
            
    # Default name coords determined for each video
    name_pos_d = {}
    height, width = frame_na.shape[:2]

    x1 = int(width * 0.256)
    x2 = int(width * 0.43)

    # P1
    y1 = int(height * 0.337)
    y2 = int(height * 0.367)
    name_pos_d[1] = [x1, x2, y1, y2]

    # P2
    y1 = int(height * 0.389)
    y2 = int(height * 0.419)
    name_pos_d[2] = [x1, x2, y1, y2]

    # P3
    y1 = int(height * 0.55)
    y2 = int(height * 0.58)
    name_pos_d[3] = [x1, x2, y1, y2]

    # P4
    y1 = int(height * 0.603)
    y2 = int(height * 0.633)
    name_pos_d[4] = [x1, x2, y1, y2]


    # SB detection region between name and score
    sb_x1 = int(width * 0.43)
    sb_x2 = int(width * 0.453)
    
    sb_upper_y1 = 0.335
    sb_lower_y1 = 0.5486



    vid_frame_total = reader.count_frames()
    vid_frame_count = 0

    try:
        frame_count_interval = round(reader.get_meta_data()['fps'])
    except:
        print('meta data error')
        frame_count_interval = 30
    
    
    while vid_frame_count + frame_count_interval <= vid_frame_total:

    #for frame_i, frame_na in enumerate(reader):
        vid_frame_count += frame_count_interval

        if vid_frame_count > 100: break
        time_vcap = time.perf_counter()

        reader._skip_frames(frame_count_interval)
        frame_na = reader._read_frame()[0]  ## or frame_na = reader.get_data(vid_frame_count)

        frame_read_l.append((time.perf_counter() - time_vcap))

        #if vid_frame_count % 30 != 0: continue  # Work on every nth frame
        #print('frame:', vid_frame_count)



        # SCOREBOARD DETECTION
        detected_colors_l = []
        current_pos = 'Upper'

        # Upper and lower SB color detection
        for y1_percent in [0.335, 0.5486]:
            sb_y1 = int(height * y1_percent)
            sb_y2 = int(sb_y1 + 1)  # Grab a row region between name and score

            sb_na = frame_na[sb_y1:sb_y2, sb_x1:sb_x2]
            sb_trans_na = sb_na[0].transpose()

            # Check each color individually
            sb_color_d = {
                'red': False,
                'green': False,
                'blue': False
            }
            for i, color in enumerate(sb_color_d):
                if all(140 > xxx > 100 for xxx in sb_trans_na[i].tolist()):  # Find which colors are within range
                    sb_color_d[color] = True

            for k, v in sb_color_d.items():
                if v:
                    #print(current_pos, 'color detected:', k)
                    detected_colors_l.append(k)

            current_pos = 'Lower'

        # Blue and red should be detected once. Green shouldn't
        if detected_colors_l.count('blue') != 1 or detected_colors_l.count('red') != 1 or detected_colors_l.count('green') != 0:
            #print('SB not detected:', detected_colors_l)
            continue
        else:
            print('SB detect. Lower:', k)




        # TITLE DETECTION
        title_d = {
            1: False,
            2: False,
            3: False,
            4: False
        }

        x1_coord = int(width * 0.258)
        x2_coord = int(width * 0.281)
        tit_y_percents = (0.3666, 0.4194, 0.5819, 0.6347)

        # Loop for each player
        for player, iiii in enumerate(tit_y_percents, 1):
            y1_coord = int(height * iiii)
            y2_coord = y1_coord + 1
            tit_reg = frame_na[y1_coord:y2_coord, x1_coord:x2_coord]
            rgb_l = tit_reg.transpose()
            # If each color channel has a value over 100 then title detected
            for each_color_l in rgb_l:
                if not any(i > 100 for i in each_color_l):
                    break  ##
            else:
                title_d[player] = True
                tit_correction = int(0.0083 * height)
                name_pos_d[player][2] -= int(0.0083 * height)
                name_pos_d[player][3] -= int(0.0083 * height)

        if any(value for value in title_d.values()):
            print('Title detected for players:', str([k for k,v in title_d.items() if v])[1:-1])



        # PARTY SYMBOL DETECTION
        party_d = {
            1: False,
            2: False,
            3: False,
            4: False
        }

        x_coord = int(width * 0.265)

        party_y_percents = (0.347, 0.401, 0.563, 0.616)

        # Loop for each player
        for player, each_p in enumerate(party_y_percents, 1):
            y_coord = int(height * each_p)

            party_red, party_green, party_blue = frame_na[y_coord, x_coord]  # Get RGB values where party symbol might be

            # Target color values for party symbol
            party_rt = 215
            party_gt = 255
            party_bt = 152
            party_l = ((party_red, party_rt), (party_green, party_gt), (party_blue, party_bt))
            # If each color channel has a value within range then symbol detected
            for party_pair in party_l:
                current, target = party_pair
                if not target * .85 < current < target * 1.15:
                    break
            else:
                party_d[player] = True
                name_pos_d[player][0] += int(0.01875 * width)

        if any(value for value in party_d.values()):
            print('Party symbol detected for players:', str([k for k,v in party_d.items() if v])[1:-1])



        # OCR
        for player, coords in name_pos_d.items():
            x1_coord, x2_coord, y1_coord, y2_coord = coords
            name_na = frame_na[y1_coord:y2_coord, x1_coord:x2_coord]

            save_img(name_na)

            name = pytesseract.image_to_string(name_na, timeout=5, config='''-c tessedit_do_invert=1 -c tessedit_char_whitelist="!\\"#$%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ " --psm 7 --oem 1''').strip()

            if name:
                add_names_f(name, video_path, name_d)


    # Write results at end of each video
    write_res_f(name_d)












print('frame read ave:', sum(frame_read_l) / len(frame_read_l))














