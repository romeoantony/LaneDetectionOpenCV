# Lane Line Detection

## Description
This project implements a lane line detection system using OpenCV and Python. The system processes a video input, detects lane lines in each frame, and outputs a new video with the detected lane lines highlighted. This can be particularly useful for applications in autonomous driving and advanced driver-assistance systems (ADAS).

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [Contact Information](#contact-information)
- [Acknowledgments](#acknowledgments)

## Features
- Detects lane lines in video frames.
- Outputs a new video with highlighted lane lines.
- Supports reading and writing images and videos using OpenCV.
- Customizable parameters for color detection and Hough Transform.

## Requirements
- Python 3.x
- OpenCV
- NumPy

You can install the required libraries using pip:

```bash
pip install opencv-python numpy
```

## Usage
1. Place your input video file (e.g., `input3.mp4`) in the project directory.
2. Run the main script:

```bash
python main.py
```

## How It Works
The program processes the video by:
- Reading each frame and resizing it.
- Converting the frame from BGR to HSV color space to detect yellow and white colors.
- Applying a mask to isolate the lane lines.
- Defining a region of interest (ROI) to focus on the lane area.
- Using Gaussian blur, dilation, and erosion to enhance the lane line features.
- Applying the Hough Transform to detect lines in the processed image.
- Drawing the detected lines on the original frame.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## Contact Information
For any inquiries, please contact:  [Akshay](mailto:romeoantony1999@gmail.com)

## Acknowledgments
- OpenCV for computer vision capabilities.
- NumPy for numerical operations.