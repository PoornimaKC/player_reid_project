# sort.py
import numpy as np
from filterpy.kalman import KalmanFilter

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.track_id_count = 1

    def update(self, dets):
        self.frame_count += 1
        trackers = []

        for det in dets:
            trackers.append([*det[:4], self.track_id_count])
            self.track_id_count += 1

        print(f"[SORT] Frame {self.frame_count}: {len(trackers)} tracks")
        return trackers
