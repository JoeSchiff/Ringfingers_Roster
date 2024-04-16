
# Desc: Extract PSN names from Rocket League videos

# cd '/media/joepers/joerassic_park_f/ps4_captures/all_kb_rl_until_6_20/Rocket League®/'
# py /home/joepers/code/get_names_rl.py *.mp4


# to do:
# improve name reading
#   remove ampersand in name - 
#   only add names if 4 are visible + replace with if not name: continue ?
#   end of game name positions are diff
#   name pos is diff if player has a title. also if in party
# detect nameplate
#   blue or orange can be on top!
# detect title
# change selectors from abs pixels to percent
# double confirm names?
# error catch
#   test file path


# 10:00 = 500 mB
# 53 Gb = 2h 50m
# 1060 minutes of footage in 170 minutes
# ~ 6.2x speed at 30 fps, sampling every other second


from datetime import datetime

from sys import argv
from imageio import get_reader
from PIL import Image, ImageOps
from pytesseract import pytesseract
import json



startTime = datetime.now()

ignore_l = ['KyleMcButt', 'TheDankyKang']




name_d = {}



# Add names to dict
def add_names_f(temp_l):
    print('addinggg', temp_l)
    for name in temp_l:
        if any(ignore_name in name for ignore_name in ignore_l): continue  # Exlude any names from the ignore list

        if name in name_d:  # name already in dict
            name_d[name].append(filename)
        else:  # new entry
            name_d[name] = [filename]



for image_path in argv[1:]:

    try:

        filename = image_path.split('/')[-1]
        reader = get_reader(image_path)
        print('\n Video:', filename)

        # Loop video by each frame
        for frame_i, im_na in enumerate(reader):

            if frame_i % 30 != 0: continue  # Work on every 60th frame

            #print('Frame number:', frame_i)
            #if frame_i > 10000: break

            temp_l = []


            # Convert frame from numpy array to image
            orig_img = Image.fromarray(im_na)

            # Invert for black text
            orig_img = ImageOps.invert(orig_img)


            # Convert to black and white for Blue team
            thresh = 170
            fn = lambda x : 255 if x > thresh else 0
            full_img = orig_img.convert('L').point(fn, mode='1')
            #full_img.show()

            # First name
            img = full_img.crop((335,240,600,265))
            #img.show()
            name = pytesseract.image_to_string(img).strip()  # Remove nonprintable form feed char and newline
            if name: temp_l.append(name)

            # Second name
            img = full_img.crop((335,273,600,295))
            #img.show()
            name = pytesseract.image_to_string(img).strip()
            if name: temp_l.append(name)


            # Convert to black and white for Orange team
            thresh = 150
            fn = lambda x : 255 if x > thresh else 0
            full_img = orig_img.convert('L').point(fn, mode='1')

            # Third name
            img = full_img.crop((335,390,600,412))
            #img.show()
            name = pytesseract.image_to_string(img).strip()
            if name: temp_l.append(name)

            # Fourth name
            img = full_img.crop((335,425,600,450))
            #img.show()
            name = pytesseract.image_to_string(img).strip()
            #name = pytesseract.image_to_string(img, timeout=5, config='''-c tessedit_do_invert=0 -c tessedit_char_whitelist="!\\"#$%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ " --psm 7 --oem 1''').strip()
            if name: temp_l.append(name)


            # Add names to dict
            if len(temp_l) == 4:
                add_names_f(temp_l)

    except Exception as errex:
        print(errex)


    print(name_d)


json_results = json.dumps(name_d, indent=4, ensure_ascii=False)
with open('/home/joepers/Desktop/rl_names', 'w', errors='replace') as name_file:
    name_file.write(json_results)




duration = datetime.now() - startTime
print(duration)










