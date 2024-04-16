# Advanced Usage

### Table of Contents
* [Manual installation method](#Manual-installation-method)
* [Options](#Options)
* [Input](#Input)
* [Output](#Output)
* [Features](#Features)
* [Performance](#Performance)
* [Tips and troubleshooting](#Tips-and-troubleshooting)
* [How it works](#How-it-works)



<br/><br/>
## Manual installation method
1\. Install Python version 3.6+. [Available here](https://www.python.org/downloads/)

2\. Install the following Python packages: pytesseract, opencv-python\
Example:\
`pip install pytesseract opencv-python-headless`

3\. Install Tesseract-OCR (version 5.0+ is strongly recommended)\
Windows: [Available here](https://github.com/UB-Mannheim/tesseract/wiki)\
Linux: [Available here](https://tesseract-ocr.github.io/tessdoc/Home.html#binaries)

I used the following to easily install Tesseract 5.0 on Ubuntu:
```
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt-get install tesseract-ocr
```

4\. Download the latest RR script located at: [ds3_rr_v0.2.0-beta.py](https://github.com/JoeUgly/Ringfingers_Roster/releases/download/ds3_rr_v0.2.0-beta/ds3_rr_v0.2.0-beta.py)

5\. Add the Tesseract executable to your PATH or state it in the RR python script (variable name `pytesseract.tesseract_cmd`)




<br/><br/>
To run the program, invoke the Python script from a CLI along with the folder(s) containing your videos.\
Example:\
`python C:\Users\Jugly\Downloads\ds3_rr.py C:\Users\Jugly\Videos\DS3\`

If no video folder is given, then it defaults to the folder containing the RR script.


<br/><br/>
## Options
The following can be given as arguments:

`--nonrecursive` &emsp;&emsp; Do not search directories recursively. [See Input](#Input)\
`--noskip` &emsp;&emsp;&emsp;&emsp;&emsp;Do not skip files if they have the same filename. [See Skip duplicates](#Skip-duplicates)\
`--strict`&thinsp;&emsp;&emsp;&emsp;&emsp;&emsp;Attempt name extraction only if a prefix and suffix are found. [See Readme](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/README.md#Misinterpreted-characters-in-prefix-or-suffix)\
`--lenient`&thinsp;&ensp;&emsp;&emsp;&emsp;&emsp;Attempt name extraction if both the prefix and suffix are missing. [See Readme](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/README.md#Misinterpreted-characters-in-prefix-or-suffix)\
`--output=`&thinsp;&ensp;&emsp;&emsp;&emsp;&emsp;Specify the location of the output (result) file. [See Output](#Output)



<br/><br/>
## Input

This program will accept any combination of these four types of arguments:
1. Options
2. A Result file used for the [Resumption feature](#Resumption)
3. A video file
4. A directory containing videos (and result files)

All options must begin with `--`.\
All result files must begin with `ds3_rr_results`.

If no files or directories are given, the program will use the directory containing the executable/script.


<br/><br/>
Directory searches are recursive by default. Disable this with the `--nonrecursive` option.\
Only directories given AFTER the --nonrecursive option will be nonrecursive.

Example:\
`python C:\Users\jhalb\Downloads\ds3_rr.py C:\Users\jhalb\folder_1\ --nonrecursive C:\Users\jhalb\folder_2\`\
Only folder_1 will be searched recursively.


<br/><br/>
## Output
This program outputs the results in JSON / Python dictionary format. The key is the player name and the value is a list of file paths (the videos in which the player's name was found).

Example result (output) file:
```
{
"Motion_1": ["/path/to/video1.mkv"],

"Noob_Slayer_2": ["/path/to/video3.mp4", "/path/to/video1.mkv", "/path/to/video4.mkv"]
}
```


<br/><br/>
To specify the output file location use the `--output` option.\
Example:\
`python ds3_rr.py /path/to/videos/ --output=/path/to/documents/`

If no location is specified, then the video directory will be used.\
If no video directory is specified, then the directory containing the Python script will be used for input and output.

**This program will overwrite the output file without warning.**

<br/><br/>
## Features

#### Resumption
This program saves its progress to a file after each video is processed. This allows for a Resumption feature which serves three purposes:\
1. If an error occurs, you don't need to start over from the beginning.\
2. When you add new videos and run this program again, it will skip videos which have already been processed.\
3. Merge the previous result file to create a single combined result file with the new additions.

To use this Resumption feature:\
Place the result file in the same directory as the videos you are processing.

Alternatively, you can explicitly give the absolute path of the result file as an argument.\
Example:\
`python ds3_rr.py /path/to/videos/ /path/to/ds3_rr_results.txt`

The results filename must begin with "ds3_rr_results".\
You can [merge multiple result files](#Merge-multiple-result-files-together).


<br/><br/>
#### Skip duplicates
By default this program will skip videos with the same filename, even if they are in different folders.\
Example (the second video will be skipped):\
C:\Users\jhalb\Desktop\video_1.mkv\
C:\Users\jhalb\Videos\DS3\pvp\video_1.mkv	

This is done to allow use of symlinks.

To skip videos only when they have the same name and location invoke the `--noskip` option.
Use this option only when different videos are expected to have the same filename.


<br/><br/>
## Performance
#### Startup
The startup process of the frozen executable may take several seconds.\
This is because the contents of the file must be written to disk as a temporary directory and then read. I may switch from Pyinstaller to Pyoxidizer to speed this up.

#### Processing
Videos consist of a lot of data, so processing it will be somewhat slow.\
The speed at which this occurs is mostly dependent on your CPU and the video bitrate. It seems that the codec / file type also plays a role. More testing is needed.

Surprisingly, the biggest bottleneck in this program is opening and reading a frame. On my Ryzen 2200G, just reading the frame data is much slower than playback speed. I suspect this is due to a lack of hardware acceleration.

[Check here for typical processing speeds](https://github.com/JoeUgly/Ringfingers_Roster/discussions/3)


<br/><br/>
**To speed up processing:**\
Move half of your videos to a different directory. Run an instance of this program on each directory.\
Example:\
`python3 ds3_rr.py /path/to/dir_1`

(Simultaneously in another shell)\
`python3 ds3_rr.py /path/to/dir_2`

You can increase the number of directories and instances until your CPU usage is maximized.

This will create two separate result files. See [Merge result files](#Merge-multiple-result-files-together)


<br/><br/>
I have made every effort to make this program as performant as possible. Including:
* Multithreading using a queue for the frames
* Reading the fewest number of frames needed to sample an "opaque" nameplate
* Using the index operator to discard color data prior to var assignment
* Cropping the Numpy array without converting to an image
* Checking for nameplate presence prior to OCR
* Single thread Tesseract (10x faster OCR on 1080p)
* Skipping duplicate files

If you have any suggestions, please let me know.



<br/><br/>
## Tips and troubleshooting

### Whitespace in file path
If there is a space character in your filename or path, then surround it with quotes.\
Example:\
`.\ds3_rr.exe DS3Videos\ --output="C:\Users\name with spaces\Desktop"`


<br/><br/>
### Old GLIBC Linux error
This program is expected to work on Ubuntu 20.04 and newer. Similar distros should work, but have not been tested.\
If you get an error similar to this:
`[13268] Error loading Python lib '/tmp/_MEIKPWmtg/libpython3.8.so.1.0': dlopen: /lib/x86_64-linux-gnu/libm.so.6: version 'GLIBC_2.29' not found (required by /tmp/_MEIKPWmtg/libpython3.8.so.1.0)`\
It probably means your distro is too old. Either upgrade, use the manual installation method, or let me know and I could make a version for you.


<br/><br/>
### Checksums
If you get an error similar to `Failed to extract ... decompression resulted in return code`
Compare the values from the checksum file to make sure your file is not corrupted. A checksum file is included in each release.

Example using Powershell:\
`Get-FileHash ds3_rr.exe`\
Example using CMD:\
`certutil -hashfile ds3_rr.exe sha256`\
Example using Bash:\
`sha256sum ds3_rr`


<br/><br/>
### Linux Permission Denied
Be sure to allow executing file as a program. (Right click on file, properties)


<br/><br/>
### Merge multiple result files together
Create an empty directory. Move the executable and the result files into the directory. Run the program.



<br/><br/>
## How it works

The program begins by putting all the videos to be processed into a list.

Two threads are started. One thread extracts frames from the video and puts the frames into a queue. The other thread takes a frame from the queue and processes it.

**Frame extraction thread**\
This thread iterates over every video in the list. Videos are read using CV2 (OpenCV) and frames are stored as Numpy arrays.\
Every 67th frame (assuming a 60 fps video) is extracted because that is the minimum sampling rate required to guarantee nameplate recognition. The nameplate appears three times, fading to transparent in between. This program aims to select a frame containing the nameplate at least one of the three times when it is completely opaque. Choosing a less frequent sampling rate would increase performance, but decrease the quality of the results.

**Frame processing thread**\
After a frame is taken from the queue, it is cropped to a small area near the top of the nameplate. The average brightness of this small area is determined. If the brightness is not within the expected range, then it is assumed that a nameplate message is not present and the frame is discarded.

If a nameplate is determined to likely be present, then the nameplate is cropped where the text is located.

Optical character recognition (OCR) using PyTesseract (Tesseract) is performed.

If an appropriate phrase is detected (such as "Invaded the World of ...", "Phantom ... has died", etc), then the player name is appended to the result (output) dictionary.

Every video name is appended to a special key in the result dictionary called `"  ALL  "`. This is done to save the current progress and to skip videos if this program is ran again.

Another frame is taken from the queue and the cycle repeats.

When both threads conclude, the results and stats are displayed.


<br/><br/>




















