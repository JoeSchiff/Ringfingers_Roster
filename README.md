<h1 align="center">Ringfinger's Roster</h1>
<h2 align="center">Extract other players' names from your Dark Souls 3 gameplay videos</h2>



<br/><br/>
### Table of Contents
* [Summary](#Summary)
* [Getting Started](#Getting-Started)
* [Limitations](#Limitations-and-known-bugs)
* [FAQ](#FAQ)
* [How can I help?](#How-can-I-help)
* [Advanced Usage](/Advanced_Usage.md)


<br/><br/>
## Summary
Scans through your DS3 gameplay videos and saves the name of each player who enters or leaves the world.\
The program outputs a list of player names, along with the name of each video which they were found in.

Use case:\
You know you fought this person before, but you don't want to manually look through hours of gameplay footage to find the right video(s).



<br/><br/>
## Getting Started
### Recommended method
The easiest way to run this program is to use one of the "frozen" executables.\
Windows 10 & 11: [ds3_rr_win_v0.2.0-beta.exe](https://github.com/JoeUgly/Ringfingers_Roster/releases/download/ds3_rr_v0.2.0-beta/ds3_rr_win_v0.2.0-beta.exe)\
Linux Ubuntu 20.04: [ds3_rr_linux_v0.2.0-beta](https://github.com/JoeUgly/Ringfingers_Roster/releases/download/ds3_rr_v0.2.0-beta/ds3_rr_linux_v0.2.0-beta)

These executables will contain the Python interpreter, the Python script, and all the dependencies.

Simply place the executable file in the same folder as your videos and double click to run it.


<br/><br/>
Alternatively, you can run the executable file from a CLI.\
Example:\
`C:\Users\jugly\Downloads\ds3_rr.exe "C:\Users\jugly\Desktop\DS3Videos\"`

If no video folder is given, then it defaults to the folder containing the executable file.

Running from a CLI may be mandatory on some systems (Linux).

[See more examples](https://github.com/JoeUgly/Ringfingers_Roster/blob/main/examples.md)


<br/><br/>
#### Manual method
If you are not using one of the platforms listed above, then you'll need to fulfill the prerequisites manually.\
See the [Manual installation method](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/Advanced_Usage.md#Manual-installation-method) section of the Advanced Usage document.

<br/><br/>
## Limitations and Known Bugs

#### Nameplate not caught
This program detects player names **ONLY** when the nameplate appears.

![](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/assets/nameplate.png)

This program assumes the entire nameplate animation (the nameplate message fading in and out multiple times) is present in the video.\
If any part of the nameplate animation is missing from the video, then this program might not capture that player's name.

<ins> **Does not work on arena fights.** </ins>


<br/><br/>
#### Misinterpreted characters in player name
Having a computer read text from an image is inherently error prone.\
It's common for a player name to appear in the output multiple times with slight errors.\
Example:
```
{
"Player_1": ["/path/to/video1.mkv"],
"PIayer_!": ["/path/to/video1.mkv"],
...
}
```

A video resolution of 720p is the minimum needed for acceptable quality of results. 1080p is much better.

Currently limited to only the Latin alphabet / ASCII printable characters.


<br/><br/>
#### Misinterpreted characters in prefix or suffix
The name extraction consists of three steps. Recognize the prefix phrase, recognize the suffix phrase, extract the leftover name.\
Using the example message "Phantom XXsoandsoXX has been summoned":\
Prefix: Phantom\
Suffix: has been summoned\
Player name: XXsoandsoXX

Default behavior:\
An exact match for either a known prefix or suffix is required. 
If a prefix or suffix is found in the text, then name extraction will occur.

This behavior can be changed by invoking the `--strict` or `--lenient` options.\
Using `--strict` will not extract a player name, unless both a prefix and a suffix is detected.\
Using `--lenient` will extract a player name, even if both the prefix and suffix are not detected.\
Example:\
`C:\Users\jugly\Desktop\Downloads\ds3_rr.exe --lenient C:\Users\jugly\Desktop\DS3Videos\`

The `--lenient` option will allow more entries into the output file, many of which will be false positives.\
If you want to maximize the chances of detecting a player's name and you are not concerned about a bloated output file, then use this option.

Conversely, if you feel that there are too many erroneous entries in the result file, then use the `--strict` option.\
Note: A single misinterpreted character in either the prefix or suffix will cause the frame (and name) to be discarded.


<br/><br/>
#### Message too long (rare)
Some messages can be so long that they extend past the edge of the nameplate and therefore not detected. This can be alleviated with the `--lenient` option. [See below.](#Misinterpreted-characters-in-prefix-or-suffix)

Some messages consist of two lines and therefore the useful text is not in the expected location and also not detected. An option to expand the search area for text may be included in a future version.


<br/><br/>
## FAQ

#### What video formats are allowed?
Each video must have a file extension known to be a video format.\
Your video formats are also limited to what OpenCV can read.\
Tested formats: webm (vp9), mkv (h264), mp4 (mov), mov (h264), wmv (wmv2), flv (flv1)

Any file which does not begin with `ds3_rr_results` will be assumed to be a video. This program should identify errors related to trying to read a non-video file and skip to the next file.

	
<br/><br/>
#### How can I skip some files in a folder?
Use your CLI's glob / wildcard capabilities.\
Example using Windows PowerShell to select only MP4 files:\
`python C:\Users\jhalb\Downloads\ds3_rr.py $(dir C:\Users\jhalb\Videos\*.mp4)`

Example using Bash to select only MP4 files:\
`python3 /home/jhalb/code/ds3_rr.py /home/jhalb/videos/*.mp4`


<br/><br/>
#### Why is it so slow?
See the [Performance section](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/Advanced_Usage.md#Performance) in the Advanced Usage document.


<br/><br/>
#### Other questions
Check out the [Advanced Usage document](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/blob/main/Advanced_Usage.md)\
or [open an issue](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/issues)


<br/><br/>
## How can I help?
Please report the following:\
Bugs, errors, problems by [starting an issue](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/issues)\
Your PC specs and performance (the last few lines of the program's CLI output). [Post here](https://github.com/JoeUgly/Dark_Souls_3_Roll_Call/discussions/3)\
Any other questions, requests, suggestions, comments, etc


<br/><br/>






