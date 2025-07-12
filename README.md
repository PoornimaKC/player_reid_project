# player_reid_project
Player Re-Identification in Soccer Footage using YOLOv11 + SORT Tracker
# Player Re-Identification Project

## Problem Statement
Detect and track soccer players consistently in video footage — ensuring each player retains a unique ID even when they leave and re-enter the frame.

---

## Analytical Goal

- Use a pre-trained YOLOv11 model for player detection.
- Track players frame-to-frame using a simple SORT tracker.
- Output a video with visible player bounding boxes and unique IDs.

---

## Input Details

- Input video: `15sec_input_720p.mp4`
- YOLOv11 weights: `best.pt`

---

##  Output Deliverables

- `tracked_output.mp4` with bounding boxes & IDs.
- Jupyter notebook: `player_reid_notebook.ipynb` showing experiments.
- `track.py` and `sort.py` source code.
- `README.md` and `REPORT.md` explaining approach & challenges.

---

##  Setup Instructions

1. **Create Conda environment**
   ```bash
   conda create -n player_reid python=3.10
   conda activate player_reid
2. **Install dependencies**
   pip install ultralytics opencv-python filterpy
3. **Run tracking**
   python track.py
Output video will be saved as tracked_output.mp4.

**Limitations**
Simple SORT version used: assigns new IDs frequently due to lack of Kalman filtering and IOU matching.

For production, appearance features or DeepSORT should be used for robust re-identification.
2
