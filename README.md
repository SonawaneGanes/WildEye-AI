# 🦌 Wild Eye AI – Wildlife Detection Result Viewer

## 📌 Project Overview

**Wild Eye AI** is a Python-based **Dash web application for reviewing and visualizing wildlife object-detection results alongside video footage**.

The application allows users to select wildlife footage and a display mode, play the video, and view object-detection information corresponding to the current video frame.

The application uses **pre-recorded object-detection results stored in CSV files**. It does **not run an object-detection model itself**; instead, it loads previously generated detection data and provides an interactive interface for analysis and visualization.

---

## 🎯 Objective

The main objective of this project is to make wildlife detection results easier to review and understand by combining:

* Wildlife video footage
* Pre-recorded object-detection results
* Confidence scores
* Object-count visualization
* Confidence heatmaps
* Interactive Dash components

This type of system can help researchers and developers inspect the output of wildlife detection systems more conveniently.

---

## 🚀 Features

* 🎥 Video playback for selected wildlife footage
* 🐾 Review of object-detection results
* 📊 Detection-score visualization
* 🍩 Object-count pie charts
* 🔥 Confidence-score heatmaps
* 🎚️ Confidence threshold filtering
* 🔄 Frame-based detection analysis
* 📋 CSV-based detection data processing
* 🔔 Notifications/help popup
* 🌐 Interactive web interface using Plotly Dash

---

## 🧠 How the Application Works

```text
                    Wildlife Video
                         |
                         v
                  Video Playback
                         |
                         v
                Current Playback Time
                         |
                         v
               Convert Time → Frame
                         |
                         v
                Detection CSV Data
                         |
                         v
              Filter by Current Frame
                         |
               +---------+---------+
               |         |         |
               v         v         v
          Score Graph   Pie Chart  Heatmap
               |         |         |
               +---------+---------+
                         |
                         v
                 Interactive Results
```

---

## 📂 Project Structure

The exact workspace structure may contain additional files, but the application expects a structure similar to:

```text
Wildeye AI/
│
├── app.py
│
├── data/
│   ├── FarmDroneDetectionData.csv
│   ├── Zebra_object_data.csv
│   ├── james_bond_object_data.csv
│   └── ...other detection CSV files
│
├── assets/
│   └── ...optional Dash CSS, images, and static files
│
├── requirements.txt
│
└── ...other project files
```

The CSV paths used by the application are resolved relative to the **current working directory**, so the application should normally be started from the project directory.

---

## 🛠️ Technology Stack

| Technology  | Purpose                               |
| ----------- | ------------------------------------- |
| Python      | Application development               |
| Dash        | Web application framework             |
| Plotly      | Data visualization                    |
| Pandas      | CSV/data processing                   |
| NumPy       | Numerical processing                  |
| Dash Player | Video playback                        |
| Flask       | Web framework dependency used by Dash |

---

## 📦 Dependencies

The original project environment specifies the following package versions:

```text
dash==0.35.2
dash-auth==1.1.2
dash-html-components==0.13.5
dash-core-components==0.43.0
dash-renderer==0.17.0
gunicorn==19.9.0
plotly==3.6.0
pillow==5.4.1
Flask==1.0.1
scipy==1.2.1
numpy==1.16.1
pandas==0.24.1
dash-player==0.0.1
```

These are historical dependency versions from the original project environment. Newer Python versions may require updated, compatible package versions.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd "Wildeye AI"
```

### 2. Create a virtual environment

#### Linux / macOS

```bash
python -m venv .venv
source ./.venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

From the project directory:

```bash
python app.py
```

On Windows:

```powershell
cd "D:\Wildeye AI"
python app.py
```

The Dash application should then start and provide a local web address.

---

## 📊 Detection Data

The application expects detection data in CSV files.

The CSV files must contain at least the following columns:

```text
class_str
frame
score
```

### Column Description

| Column      | Description                                          |
| ----------- | ---------------------------------------------------- |
| `class_str` | Name of the detected object/species                  |
| `frame`     | Video frame number associated with the detection     |
| `score`     | Detection confidence score, normally between 0 and 1 |

Example:

```csv
class_str,frame,score
zebra,120,0.94
lion,120,0.87
zebra,121,0.91
lion,122,0.89
```

---

## 🔄 Main Workflow

### 1. Load Detection Data

The application loads configured CSV files using Pandas.

### 2. Load Footage

`load_all_footage()` loads the available detection data and creates mappings between footage and the corresponding video URLs.

### 3. Select Footage

The user selects the required footage from the dropdown.

### 4. Select Display Mode

The user chooses how the detection information should be visualized.

### 5. Track Video Time

The video player provides the current playback time.

### 6. Convert Time to Frame

The application converts playback time into a video frame using:

```text
frame = playback_time × FRAMERATE
```

The configured frame rate is:

```python
FRAMERATE = 24.0
```

Therefore, the frame rate must match the source video and the frame numbering used by the CSV data.

### 7. Filter Detection Results

The application selects detections belonging to the current frame and applies the selected confidence threshold.

### 8. Generate Visualizations

The filtered detection data is used to generate graphs such as:

* Detection-score graph
* Object-count pie chart
* Confidence heatmap

---

## 🔧 Core Functions

### `load_data(path)`

Reads a detection CSV using Pandas.

It:

* loads the data
* counts object classes
* prepares class information for visualization
* creates data required by the heatmap
* returns the processed values and original DataFrame

---

### `load_all_footage()`

Loads the configured detection CSV files into memory.

It also creates URL mappings for:

* normal footage
* bounding-box footage

This function runs during application startup.

---

### `markdown_popup()`

Creates the Notifications popup containing explanatory information for the user.

---

### `select_footage(footage, display_mode)`

Selects the appropriate video URL according to the selected footage and display mode.

---

### `update_click_output(button_click, close_click)`

Controls opening and closing of the Notifications popup based on button-click counts.

---

### `update_output(dropdown_value)`

Controls the selected visualization view and displays the corresponding graph.

---

### `update_detection_mode(value)`

Changes the detection-score visualization based on the selected mode.

---

### `update_score_bar(...)`

This callback:

1. receives the current video time
2. converts the time into a frame number
3. finds detections for that frame
4. applies the confidence threshold
5. plots detection scores
6. limits the displayed results to a maximum of eight detections

---

### `update_object_count_pie(...)`

Counts detected objects/classes for the current frame and displays them using a pie chart.

---

### `update_heatmap_confidence(...)`

Displays class-wise confidence information using a heatmap.

Class names are used as annotations so that confidence values can be inspected visually.

---

## 🎥 Video and Frame Synchronization

The application uses:

```python
FRAMERATE = 24.0
```

to map video playback time to frame numbers.

For example:

```text
Playback time = 5 seconds

Frame = 5 × 24
      = 120
```

Therefore, detection data for frame `120` is displayed when the video reaches approximately 5 seconds.

The source footage and CSV data must use compatible frame-rate and frame-numbering assumptions.

---

## 📈 Visualizations

### Detection Score Graph

Shows confidence scores for objects detected in the current frame.

The confidence threshold can be adjusted to remove low-confidence detections.

---

### Object Count Pie Chart

Displays how many detections belong to each object class in the selected frame.

Example:

```text
Lion   → 2
Zebra  → 5
```

---

### Confidence Heatmap

Displays confidence scores for detected classes using a heatmap representation.

This helps the user compare detection confidence among different classes.

---

## 🎬 Available Footage

The application currently exposes selected footage in its footage dropdown, including:

* Farm Drone
* Lion Fighting Zebras

Additional footage can be loaded by adding the corresponding CSV configuration and video mapping.

---

## ⚠️ Important Implementation Notes

### The application does not run the object-detection model

This is an important architectural detail.

The application consumes **pre-recorded detection results** from CSV files.

```text
Object Detection Model
        ↓
Detection Results
        ↓
CSV files
        ↓
Wild Eye AI Dash Application
        ↓
Visualization / Review
```

The actual object-detection model is outside the scope of this Dash application.

---

### Current Frame Rate

```python
FRAMERATE = 24.0
```

This value must match the video and detection-data generation process.

---

### Debug Mode

The application currently uses:

```text
DEBUG = True
```

This enables Dash debugging and additional startup information.

For production deployment, debug mode should normally be disabled.

---

### Relative Import Note

The source code contains:

```python
from .exceptions import ObsoleteAttributeException
```

This is a relative import.

Running:

```bash
python app.py
```

directly may fail if `app.py` is not being executed as part of a Python package or if the required `exceptions` module is unavailable.

Verify whether this import is required by the actual project before deployment.

---

## 🌍 Applications

This type of application can be used for:

* Wildlife research
* Camera-trap result analysis
* Wildlife monitoring
* Object-detection model evaluation
* Conservation data review
* Computer-vision experiment analysis
* Video-based detection auditing

---

## 🔮 Future Improvements

Possible improvements include:

* Integrating a live object-detection model
* Supporting uploaded videos
* Supporting additional wildlife species
* Adding real-time detection
* Adding bounding-box overlays
* Adding detection statistics across complete videos
* Adding CSV upload through the UI
* Adding downloadable reports
* Adding database support
* Deploying the application using Docker
* Adding authentication and user management
* Adding cloud-based storage

---

## 🧪 Testing

Recommended testing areas:

### Data Tests

* Verify required CSV columns exist.
* Verify frame values are valid.
* Verify confidence scores are between 0 and 1.
* Verify class names are valid.

### Visualization Tests

* Correct frame is selected.
* Confidence filtering works.
* Pie chart counts are correct.
* Heatmap values correspond to detection data.

### Video Tests

* Footage URLs are valid.
* Frame rate matches detection data.
* Playback time correctly maps to frame number.

---

## 💡 Example

Suppose the user watches a video at:

```text
Time = 10 seconds
```

With:

```text
FRAMERATE = 24 FPS
```

The application calculates:

```text
10 × 24 = Frame 240
```

It then searches the detection CSV for:

```text
frame = 240
```

and visualizes the detections found for that frame.

---

## 📌 Project Highlights

* Built a Python-based Dash application for reviewing wildlife object-detection results.
* Integrated video playback with frame-based detection analysis.
* Processed prerecorded detection data using Pandas and NumPy.
* Created interactive Plotly visualizations for detection confidence and object counts.
* Implemented confidence-threshold filtering.
* Developed class-wise confidence heatmap visualization.
* Built an end-to-end review workflow from video playback to detection visualization.

---

## 👨‍💻 Tech Stack

```text
Python
Dash
Plotly
Pandas
NumPy
Dash Player
Flask
Pillow
SciPy
Gunicorn
```

---

## 📜 License

Add the license applicable to your repository.

For example:

```text
MIT License
```

---

## ⭐ Project Summary

**Wild Eye AI** is an interactive wildlife detection result analysis platform built with Python and Dash. It connects wildlife video playback with pre-recorded object-detection data and provides visual tools for examining detection confidence, object counts, and class-level detection behavior.

The project is designed primarily as a **detection-result review and visualization system**, while the actual object-detection model can be treated as a separate upstream component.
