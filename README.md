# Face Recognition Attendance System

## Overview
This project is a face recognition-based attendance system that uses Dlib for facial detection and recognition, integrated with a SQLite database for attendance tracking. The system includes a GUI for face registration, real-time attendance marking with voice feedback, and feature extraction for face recognition. It supports video input from a camera and generates attendance logs. Additionally, it features an eye tracking visualization that now tracks the person's face with a digital eye representation.

## Features
- **Face Detection & Recognition**: Uses Dlib's frontal face detector, landmark predictor, and ResNet50 model for 128D face descriptors.
- **Attendance Tracking**: Marks attendance in a SQLite database with timestamps, announced via text-to-speech (gTTS).
- **Real-Time Processing**: Captures video from a webcam, processes frames, and identifies individuals.
- **Face Registration**: GUI-based interface (Tkinter) to register new faces and save them for recognition.
- **Feature Extraction**: Extracts and averages 128D face features, saved to a CSV file for recognition.
- **Voice Feedback**: Announces recognized names in Hindi or English using gTTS.
- **Eye Tracking with Face Detection**: Visualizes digital eyes that move based on the detected face's position using Pygame and OpenCV.
- **Logging**: Detailed logs for debugging and monitoring system performance.

## Prerequisites
- Python 3.7+
- OpenCV (`cv2`)
- Dlib (with pre-trained models)
- gTTS (Google Text-to-Speech)
- NumPy
- Pandas
- SQLite3
- FFmpeg (for audio playback)
- Tkinter
- PIL (Python Imaging Library)
- Pygame

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd face-recognition-attendance-system
   ```

2. **Set Up Environment**:
   Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download Dlib Models**:
   - Place `shape_predictor_68_face_landmarks.dat` and `dlib_face_recognition_resnet_model_v1.dat` in `data/` directory (paths as specified in code).

4. **Run the Application**:
   - Start face registration: `python get_faces_from_camera_tkinter.py`
   - Start attendance system: `python voice.py`
   - View attendance: `python app.py`
   - Run eye tracking: `python eye.py`

## Project Structure
- **`voice.py`**: Main script for real-time face recognition and attendance marking with voice output.
- **`eye.py`**: Pygame-based visualization of face tracking with digital eyes that follow the detected face.
- **`app.py`**: Flask web app to view attendance records.
- **`get_faces_from_camera_tkinter.py`**: Tkinter GUI for registering new faces.
- **`features_extraction_to_csv.py`**: Extracts and saves face features to `features_all.csv`.
- **`requirements.txt`**: List of names (to be replaced with actual dependency list).
- **`face_detection logs.txt`**: Log file with execution details.
- **`data/`**: Directory for Dlib models, face images, and CSV features.

## Usage

### Face Registration
- Run `get_faces_from_camera_tkinter.py` to open the GUI.
- Capture faces using the camera; enter names and save images to `data/data_faces_from_camera/`.
- Extract features by running `features_extraction_to_csv.py`.

### Attendance System
- Run `voice.py` to start the attendance system.
- The camera detects faces, matches them with registered features, and logs attendance in `attendance.db`.
- Recognized names are announced (e.g., "नमस्ते, aditya").

### Web Interface
- Run `app.py` to launch a Flask app at `http://localhost:5000`.
- Select a date to view attendance records.

### Eye Tracking
- Run `eye.py` to launch a Pygame window.
- The digital eyes will track the movement of the detected face from the webcam input.

## Configuration
- **Database**: Attendance is stored in `attendance.db` with a table named `attendance`.
- **Paths**: Update model paths in scripts if different from `/home/cpatwadityasharma/attendence/...`.
- **Language**: Modify `speak()` function in `voice.py` to change language (default: Hindi).
- **Today's Date and Time**: The system is operational as of 12:29 PM IST on Wednesday, August 20, 2025.

## Notes
- Ensure FFmpeg is installed for audio playback (`ffplay`).
- The system assumes a webcam at index 0; adjust in `voice.py` or `eye.py` if needed.
- Logs are saved in `face_detection logs.txt` (example date: 25/03/2025).
- Recognized names (e.g., "aditya") must match those in `features_all.csv`.
- The eye tracking in `eye.py` uses face detection to adjust pupil position dynamically.

## Troubleshooting
- **No Face Detected**: Check camera connection or lighting conditions.
- **Audio Issues**: Verify FFmpeg installation and audio permissions.
- **Model Errors**: Ensure Dlib models are correctly placed.
- **Database Errors**: Verify `attendance.db` is writable.
- **Eye Tracking Issues**: Ensure OpenCV is properly configured and camera is accessible.

## Contributing
Contributions are welcome! Submit issues or PRs for bug fixes, UI improvements, or additional features.

## License
MIT License – see [LICENSE](LICENSE) for details.
