# Center Stage with Face and Motion Detection

This project is an implementation of a **Center Stage-like feature** using **OpenCV** and **Mediapipe**. The application uses face detection to dynamically center the camera view on detected faces and switches to motion detection when no faces are found, ensuring a focused and responsive camera framing.

## Features

- **Face Detection**: Detects faces in real-time and adjusts the frame to center on them.
- **Motion Detection**: Tracks motion when no faces are present to maintain focus on significant movement.
- **Smooth Transitions**: Implements smooth cropping to avoid sudden jumps in framing.
- **Dynamic Cropping**: Ensures the frame is adjusted symmetrically while maintaining space above the head for a natural look.

## Technologies Used

- **Python**
- **OpenCV**: For image and video processing.
- **Mediapipe**: For efficient and robust face detection.
- **Threading**: To perform face detection asynchronously for improved performance.

## Prerequisites

Before running the project, ensure you have the following installed:

1. Python (>= 3.7)
2. Required Python libraries:
   - `opencv-python`
   - `mediapipe`

Install the libraries using pip:
```bash
pip install opencv-python mediapipe
