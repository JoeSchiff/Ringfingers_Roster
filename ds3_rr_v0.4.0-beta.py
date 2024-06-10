version = "0.4.0-beta"


import time
import sys
import os
import ntpath
import imageio.v3 as iio
import mimetypes
from pytesseract import pytesseract
import json
import threading
import queue


# Benchmarking
startTime = time.time()
time_vcap_l = []  # Video capture
time_frame_read_l = []  # Frame read
time_crop_l = []  # Crop/misc
time_tess_l = []  # OCR

# Phrases to remove from player names
prefix_l = [
    "Phantom",
    "Blue spirit",
    "Blade of the Darkmoon",
    "Dark spirit",
    "Mad dark spirit",
    "Aldrich Faithful",
    "Loyal spirit, Aldrich Faithful",
    "Watchdog of Farron",
    "Loyal spirit, Watchdog of Farron",
    "Task completed. Blade of the Darkmoon",
    "Spear of the Church",
    "Invaded the world of",
]
suffix_l = [
    "summoned",
    "has died",
    "has returned home",
    "summoned through concord!",
    "summoned through concord",
    "invaded",
    ", disturber of sleep",
    "has returned to their world",
    "has returned to their world.",
]
no_suffix_l = [
    "Invaded the world of Host of Embers",
    "Invaded by dark spirit",
    "Invaded by mad dark spirit",
    "Summoned to the world of Host of Embers",
    "Invaded by Spear of the Church",
    "Invaded by Aldrich Faithful",
]


class Video:
    current_file_i = -1  # for prog
    skip_tally = 0  # for prog
    frame_gt = 0  # Number of frames grand total
    duration_gt = 0  # Footage duration grand total
    kbit_per_frame_l = []  # Kbits per frame grand total

    def __init__(self, path):
        self.path = path
        self.name = ntpath.basename(path)
        Video.current_file_i += 1
        consec_err = 0

    def is_video_file(self):
        mimetype_res = mimetypes.guess_type(self.name)[0]
        if not mimetype_res or not mimetype_res.startswith("video"):
            print(
                "File MIME type detected as non-video:",
                mimetype_res,
                "\n Skipping:",
                self.name,
            )
            return False
        else:
            return True

    def get_video_capture(self):
        try:
            self.frame_count = 0
            self.time_vcap = time.perf_counter()
            self.vcap = iio.imopen(self.path, "r", plugin="pyav")
            time_vcap_l.append((time.perf_counter() - self.time_vcap))
            return True
        except Exception as errex:
            print(
                "__Error: Unable to read file as video.",
                errex,
                "\nSkipping:",
                self.path,
            )
            return False

    def select_next_frame(self):
        self.frame_count += self.frame_count_interval

    def get_video_info(self):
        try:
            # self.frame_total = self.vcap.get(7)  # Num of frames in video
            # if self.frame_total < 1: raise
            metadata = iio.immeta(self.path, plugin="pyav")

            # alt: self.frame_total = self.vcap._container.streams.video[0].frames

            self.frame_rate = metadata["fps"]
            print("fps:", self.frame_rate)
            self.frame_count_interval = round(
                self.frame_rate * 1.116666667
            )  # Select every 67th frame (on 60fps)
            self.video_duration = metadata["duration"]
            self.frame_total = self.video_duration * self.frame_rate  # not accurate
            # print('video_duration:', self.video_duration)
            # self.kbit_per_frame = round(self.vcap.get(47) / self.frame_rate)  # For bitrate benchmark
            # self.weighted_kbit = self.kbit_per_frame * self.frame_total / self.frame_count_interval  ## ?
        except Exception as errex:
            print("__Error: Unable to get video info. Skipping:", self.path, errex)

    def tally_video_stats(self):
        Video.frame_gt += self.frame_total
        Video.duration_gt += self.video_duration
        # Video.kbit_per_frame_l.append(self.weighted_kbit)


class Frame:
    def __init__(self, vid):
        self.vid = vid
        time_frame_read = time.perf_counter()
        try:
            self.numpy_array = vid.vcap.read(index=vid.frame_count)  # Decode frame
            self.frame_index = vid.frame_count
            time_frame_read_l.append((time.perf_counter() - time_frame_read))
        except StopIteration:
            print("end of video:", self.vid.name)
        except Exception as errex:
            print("vcap frame read failed:", errex)

    def available(self):
        if hasattr(self, "numpy_array"):
            # print(self.vid.frame_count)
            return True


def set_default_options():
    arg_d = {
        "recursive": True,
        "noskip": False,
        "input_dir_l": [],
        "output_dir": None,
        "result_file_l": [],
        "leniency": 1,
    }
    return arg_d


def parse_option_arg(arg, arg_d):
    match arg:
        case "--nonrecursive":
            arg_d["recursive"] = False
            print("Option set: nonrecursive")

        case "--noskip":
            arg_d["noskip"] = True
            print("Option set: noskip")

        case s if s.startswith("--output="):
            arg_d["output_dir"] = "".join(arg.split("--output=")[1:])
            print("Option set: output location:", arg_d["output_dir"])

        case "--strict":
            arg_d["leniency"] = 0
            print("Option set: strict phrase matching")

        case "--lenient":
            arg_d["leniency"] = 2
            print("Option set: lenient phrase matching")

        case other:
            print("\nOption not recognized:", arg)

    return arg_d


def parse_dir_arg(arg, arg_d):
    arg_d["input_dir_l"].append(arg)

    # Gather sub dirs only when recursive
    if arg_d["recursive"]:
        child_l = [
            os.path.join(arg, child) for child in os.listdir(arg)
        ]  # List of args in dir
    else:
        child_l = [
            os.path.join(arg, child)
            for child in os.listdir(arg)
            if os.path.isfile(os.path.join(arg, child))
        ]  # List of files in dir

    arg_d = parse_args(child_l, arg_d)
    return arg_d


def parse_file_arg(arg, arg_d):
    if ntpath.basename(arg) == ntpath.basename(result_filename):  # Result file
        print("Result file detected:", arg)
        arg_d["result_file_l"].append(arg)
    else:
        all_files_l.append(arg)

    return arg_d


# Gather all files and options from args
def parse_args(arg_l, arg_d):
    for arg in arg_l:
        if arg.startswith("--"):
            arg_d = parse_option_arg(arg, arg_d)

        elif os.path.isdir(arg):
            arg_d = parse_dir_arg(arg, arg_d)

        elif os.path.isfile(arg):
            arg_d = parse_file_arg(arg, arg_d)

        else:
            print("Argument must be an option, directory, or file:", arg)

    return arg_d


def merge_input_files(result_file_l):
    for input_file in result_file_l:
        with open(input_file, "r") as res_file:
            new_d = json.loads(res_file.read())

        for key, value in new_d.items():
            if key in player_name_d:  # Player name already exists
                for each_vid in value:  # Append all videos to the key
                    if not each_vid in player_name_d[key]:  # Prevent dups
                        player_name_d[key].append(each_vid)
            else:
                player_name_d[key] = value


def check_no_input_files(arg_d):
    if not all_files_l:
        print("Using script/exe dir for input")
        arg_d = parse_dir_arg(os.path.dirname(os.path.abspath(sys.argv[0])), arg_d)

    if not all_files_l:
        print("No files detected.")
        sys.exit()


def get_output_location(arg_d):
    if not arg_d["output_dir"]:
        if len(arg_d["input_dir_l"]) == 1:
            print("Using video dir for output")
            arg_d["output_dir"] = arg_d["input_dir_l"][0]
        else:
            print("Using script/exe dir for output")
            arg_d["output_dir"] = os.path.dirname(os.path.abspath(sys.argv[0]))

    output_file_path = os.path.abspath(
        os.path.join(arg_d["output_dir"], result_filename)
    )
    return arg_d, output_file_path


# Append every video from ALL key to checked_files_l
def init_checked_list():
    for value in player_name_d["  ALL  "]:
        checked_list_entry(value)


def set_tess_exe():
    if hasattr(sys, "_MEIPASS"):
        print("Running as frozen executable")
        tess_dir = os.path.join(
            sys._MEIPASS, "tess"
        )  # Custom dir created when freezing executables

        # Path to Tesseract executable
        if "tesseract.exe" in os.listdir(tess_dir):
            print("Trying Windows Tesseract ...")
            pytesseract.tesseract_cmd = os.path.join(tess_dir, "tesseract.exe")
        elif "tesseract" in os.listdir(tess_dir):
            print("Trying Linux Tesseract ...")
            os.environ["TESSDATA_PREFIX"] = tess_dir  # Needed to detect shared objects
            pytesseract.tesseract_cmd = os.path.join(tess_dir, "tesseract")
        else:
            print("Tesseract executable cannot be found. Exiting ...")
            return False

    # Tesseract executable must be on PATH or stated explicitly
    else:
        print("Running as a script")
        ## Uncomment next line and replace with your location of the Tesseract executable
        pytesseract.tesseract_cmd = (
            r"C:\Users\jschiffler\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
        )


def tess_working():
    try:
        pytesseract.get_tesseract_version()
        lang_l = pytesseract.get_languages()
        if "eng" not in lang_l:
            raise Exception("eng.traineddata not found")
        print("Tesseract is working")
        os.environ["OMP_THREAD_LIMIT"] = "1"  # Use only one cpu core for Tesseract

    except Exception as errex:
        print("pytesseract:", errex)
        print(
            "\n\n __Error: Tesseract is not working. If you are running as a script make sure the Tesseract executable is on the PATH or explicitly stated in the RR python script."
        )
        print("Check the Advanced Usage document on Github for more info.")
        sys.exit()


def is_file_readable(filepath):
    if not os.access(filepath, os.R_OK):
        print("Error: File is not readable:", filepath)
        return False


def check_writeable_dir():
    try:
        test_loc = os.path.join(arg_d["output_dir"], "rr_test_filename")

        if not os.path.isdir(arg_d["output_dir"]):
            print("\n\n __Error: Output location must be a directory.\n\n")
            sys.exit()

        with open(test_loc, "w", errors="replace") as output_file:
            output_file.write("TEST_TEXT")

        with open(test_loc, "r") as output_file:
            content = output_file.read()

        if content == "TEST_TEXT":
            os.remove(test_loc)

        else:
            raise
    except:
        print(
            "\n\n __Error: Can not read/write at output file location. Check the path and permissions."
        )
        print("Output file location:", arg_d["output_dir"])
        sys.exit()


# Extract player names from text
def get_player_name(text):
    print(text)  ##
    name = text  # This is needed for lenient matching
    lenient_val = arg_d[
        "leniency"
    ]  # Decrement this value to allow for 1 or 2 missing phrases
    try:

        # No suffix phrases
        for prefix in no_suffix_l:
            if text.startswith(prefix):
                name = text.split(prefix, maxsplit=1)[1].strip()
                return name

        # Phantom prefix detection
        for prefix in prefix_l:
            if text.startswith(prefix):
                name = text.split(prefix, maxsplit=1)[1].strip()
                break

        else:
            print("No prefix detected:", text)
            if not lenient_val:
                return None
            lenient_val -= 1

        # Phantom suffix detection
        for suffix in suffix_l:
            if text.endswith(suffix):
                name = name.rsplit(suffix, maxsplit=1)[0].strip()
                print("Phrase match:", text)
                return name

        else:
            print("No suffix detected:", text)
            if lenient_val:
                return name  # lenient val > 0
            else:
                return None

    except Exception as errex:
        print("__Error on player name extraction:", errex)
        return None


def add_player_name(name, video_path, player_name_d):
    if name in player_name_d:
        if not video_path in player_name_d[name]:  # Prevent dups
            print("Adding video to existing name:", video_path, name)
            player_name_d[name].append(video_path)
    else:
        print("Adding video to new name:", video_path, name)
        player_name_d[name] = [video_path]


def get_json_from_dict(d):
    return json.dumps(d, indent=4, ensure_ascii=False)


def write_res(player_name_d):
    json_results = get_json_from_dict(player_name_d)
    with open(output_file_path, "w", errors="replace") as output_file:
        output_file.write(json_results)


def checked_list_entry(video_path):
    if arg_d["noskip"]:
        video_name_or_path = video_path
    else:
        video_name_or_path = ntpath.basename(video_path)

    if video_name_or_path in checked_files_l:
        print("\n Skipping file:", video_path)
        return False
    else:
        checked_files_l.append(video_name_or_path)
        return True


def debug_end_early():
    breakpoint = 3000
    if frame_total > breakpoint:
        frame_total = breakpoint
    return frame_total


# Put frames in queue
def get_frames(all_files_l):
    for video_path in all_files_l:
        print("\n Reading file:", video_path)

        vid = Video(video_path)

        try:
            if not checked_list_entry(vid.path):
                continue

            if not vid.is_video_file():
                continue

            if not vid.get_video_capture():
                continue

            vid.get_video_info()

            # frame_total = debug_end_early()

            while True:  # Loop until end of video
                frame = Frame(vid)

                if not frame.available():
                    break

                vid.select_next_frame()

                with q_lock:
                    frame_queue.put((frame, False), block=True, timeout=None)

            vid.tally_video_stats()  # End of each video

        except Exception as errex:
            print(errex, sys.exc_info()[2].tb_lineno)

    # Close thread on end of all videos
    print("\nEnd of video frames")
    with q_lock:
        frame_queue.put(("END", True), block=True)


def last_frame_detect(frame, last_frame_b):
    if last_frame_b:
        player_name_d["  ALL  "].append(
            frame.vid.path
        )  # Append completed video name to ALL key
        write_res(player_name_d)  # Save new results file
        print("\n End processing vid:", frame.vid.path)


def end_detect(frame):
    if isinstance(frame, str) and frame == "END":
        print("END msg detected")
        return True


def wait_for_frame():
    while True:
        try:
            with q_lock:
                return frame_queue.get_nowait()
        except queue.Empty:
            print("queue empty")
            time.sleep(0.5)
        except Exception as errex:
            print("__Error: queue:", errex)


# Select area above nameplate text
# Crop as percent so unaffected by resolution
def crop_background(frame):

    # time_crop = time.perf_counter()
    height, width = frame.numpy_array.shape[:2]
    x1_coord = round(width * 0.29)
    x2_coord = round(width * 0.71)
    y1_coord = round(height * 0.681)
    y2_coord = round(height * 0.695)

    return frame.numpy_array[
        y1_coord:y2_coord, x1_coord:x2_coord
    ]  # Crop Numpy array with index operator because it's faster


# Select area above nameplate text
# Crop as percent so unaffected by resolution
def crop_nameplate(frame):

    # time_crop = time.perf_counter()
    height, width = frame.numpy_array.shape[:2]
    x1_coord = round(width * 0.29)
    x2_coord = round(width * 0.71)
    y1_coord = round(height * 0.681)
    y2_coord = round(height * 0.73)

    return frame.numpy_array[
        y1_coord:y2_coord, x1_coord:x2_coord
    ]  # Crop Numpy array with index operator because it's faster


# Skip if too dark or too bright, ie: nameplate not detected
def nameplate_detect(cropped_arr):
    avg_brightness = cropped_arr.mean()
    if 11 < avg_brightness < 14:  # Actual values: 12.46, 13.12
        return True


# Get text from image, don't invert, whitelist ASCII chars, expect one line of text
def get_ocr_text(cropped_arr):
    tess_config = """-c tessedit_do_invert=1 -c tessedit_char_whitelist="!\\"#$%&\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~ " --psm 7 --oem 1"""
    t4 = time.perf_counter()

    ocr_text = pytesseract.image_to_string(cropped_arr, timeout=5, config=tess_config)

    time_tess_l.append((time.perf_counter() - t4))
    return ocr_text.strip()


# Get frames from queue and process
def process_frames(player_name_d):
    while True:
        frame, last_frame_b = wait_for_frame()

        if end_detect(frame):
            break

        try:
            display_progress(frame)

            cropped_arr = crop_background(frame)
            if not nameplate_detect(cropped_arr):
                continue

            cropped_arr = crop_nameplate(frame)
            ocr_text = get_ocr_text(cropped_arr)
            if not ocr_text:
                continue

            player_name = get_player_name(ocr_text)
            if not player_name:
                continue

            add_player_name(player_name, frame.vid.path, player_name_d)

        except Exception as errex:
            print(errex, sys.exc_info()[2].tb_lineno)

        # finally:
        # last_frame_detect(frame, last_frame_b)

    print("\nEnd of all video processing")


def display_progress(frame):
    if Video.skip_tally < 50:
        Video.skip_tally += 1
        return

    Video.skip_tally = 0

    file_prog = frame.frame_index / frame.vid.frame_total

    total_prog_inc = 1 / len(all_files_l) * 100
    additional_inc = round(total_prog_inc * file_prog)

    total_prog_simple = round(Video.current_file_i / len(all_files_l) * 100)
    total_prog_adv = additional_inc + total_prog_simple

    print("\nFile progress:", str(round(file_prog * 100)) + "%")
    print("Total progress:", str(total_prog_adv) + "%")
    print("Frames in queue:", frame_queue.qsize(), "\n")


def display_results():
    print(get_json_from_dict(player_name_d))
    print("\n\n\n\t-------- Complete. --------")
    print("\nOutput file saved at:", output_file_path)


def prevent_divide_by_zero():
    for each_l in [
        Video.kbit_per_frame_l,
        time_vcap_l,
        time_frame_read_l,
        time_crop_l,
        time_tess_l,
    ]:
        if not each_l:
            each_l.append(0)


def display_stats():
    duration = time.time() - startTime

    print("\n\nRun time duration:", round(duration / 60), "minutes")
    print("Footage processed:", round(Video.duration_gt / 60), "minutes")
    print("\nAvg processing speed:", str(round(Video.duration_gt / duration, 1)) + "x")
    print("Avg frames per sec:  ", round(Video.frame_gt / duration))
    print(
        "kbits per sec:       ", round(sum(Video.kbit_per_frame_l) / duration)
    )  ## should this be averaged?
    print("\nvideo capture avg:", round(sum(time_vcap_l) / len(time_vcap_l), 4))
    print(
        "frame read avg:   ", round(sum(time_frame_read_l) / len(time_frame_read_l), 4)
    )
    # print('time_crop_l avg:', sum(time_crop_l) / len(time_crop_l))
    print("OCR read avg:     ", round(sum(time_tess_l) / len(time_tess_l), 4))
    print(f"\nVersion: {version}")


if __name__ == "__main__":
    print("Arguments:", sys.argv[1:])

    result_filename = "ds3_rr_results.txt"
    queue_max_size = 200
    all_files_l = []
    checked_files_l = []  # Used only by get_frames to track completed videos
    player_name_d = {
        "  ALL  ": []
    }  # Used by process_frames to mark a video as complete and for Resumption

    arg_d = set_default_options()
    arg_d = parse_args(sys.argv[1:], arg_d)
    check_no_input_files(arg_d)
    merge_input_files(arg_d["result_file_l"])
    init_checked_list()
    arg_d, output_file_path = get_output_location(arg_d)
    print(get_json_from_dict(arg_d))

    set_tess_exe()
    tess_working()

    check_writeable_dir()

    q_lock = threading.Lock()
    frame_queue = queue.Queue(queue_max_size)

    p1 = threading.Thread(target=get_frames, args=(all_files_l,))
    p1.start()
    p2 = threading.Thread(target=process_frames, args=(player_name_d,))
    p2.start()

    p1.join()
    p2.join()

    display_results()

    prevent_divide_by_zero()
    display_stats()

    # input('END')
