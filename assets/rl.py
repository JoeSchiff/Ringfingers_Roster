
'''

open vid
grab frame
grab player 1 sb region
test for orange
if true:
    grab and test p2 region? unn?
    if true: grab p3 and p4 regions and test for blue

if false:
test for blue
    repeat

do we need to determine color? only for sb determination

grab and test party symbol x4 +
    party sym pos depends on title. move target pos or not? should land in symbol either way
grab and test title. x4 +

grab sb color regions x4? x2?


Default coords must be percent
'''









from imageio import get_reader
from PIL import Image
import numpy as np
from pytesseract import pytesseract
import time




image_path = '/media/joepers/joerassic_park_f/ps4_captures/all_kb_rl_until_6_20/Rocket League®/Rocket League®_20191126215201.mp4'

#im = Image.open(r"C:\Users\jschiffler\Downloads\rl.bmp")

reader = get_reader(image_path)




inc = 0
def save_img(numpy_array):
    global inc
    inc += 1
    im = Image.fromarray(numpy_array)
    nameee = '/home/joepers/Desktop/t/here' + str(inc) + '.jpeg'
    im.save(nameee)




frame_read_l = []
time_vcap = time.perf_counter()

for frame_i, frame_na in enumerate(reader):
    try:
        frame_read_l.append((time.perf_counter() - time_vcap))

        if frame_i > 1000: break
        if frame_i % 30 != 0: continue  # Work on every 60th frame
        #print('frame:', frame_i)


        # Default name coords
        height, width = frame_na.shape[:2]

        name_pos_d = {}

        x1 = int(width * 0.256)
        x2 = int(width * 0.43)

        # P1
        y1 = int(height * 0.337)
        y2 = int(height * 0.365)
        name_pos_d[1] = [x1, x2, y1, y2]

        # P2
        y1 = int(height * 0.389)
        y2 = int(height * 0.417)
        name_pos_d[2] = [x1, x2, y1, y2]

        # P3
        y1 = int(height * 0.55)
        y2 = int(height * 0.58)
        name_pos_d[3] = [x1, x2, y1, y2]

        # P4
        y1 = int(height * 0.605)
        y2 = int(height * 0.632)
        name_pos_d[4] = [x1, x2, y1, y2]





        # SCOREBOARD DETECTION
        x1_coord = int(width * 0.43)
        x2_coord = int(width * 0.453)
        detected_colors_l = []
        current_pos = 'Upper'

        # Upper and lower SB color detection
        for y1_percent in [0.335, 0.5486]:
            y1_coord = int(height * y1_percent)
            y2_coord = int(y1_coord + 1)  # Grab a row region between name and score
            sb_na = frame_na[y1_coord:y2_coord, x1_coord:x2_coord]
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



    except Exception as errex:
        print('__Error:', errex)

    finally:
        time_vcap = time.perf_counter()








    for player, coords in name_pos_d.items():
        x1_coord, x2_coord, y1_coord, y2_coord = coords
        name_na = frame_na[y1_coord:y2_coord, x1_coord:x2_coord]

        save_img(name_na)

        text = pytesseract.image_to_string(name_na, timeout=5, config='''-c tessedit_do_invert=1 -c tessedit_char_whitelist="!\\"#$%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ " --psm 7 --oem 1''').strip()

        if text: print('Name detect:', text)





print('frame read ave:', sum(frame_read_l) / len(frame_read_l))














