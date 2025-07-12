import cv2
from ultralytics import YOLO
from sort import Sort

# === Load YOLO model ===
print("[INFO] Loading YOLOv11 weights...")
model = YOLO('best.pt')

# === Load input video ===
video_path = '15sec_input_720p.mp4'
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"[ERROR] Could not open video: {video_path}")
    exit()

width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = video.get(cv2.CAP_PROP_FPS)

print(f"[INFO] Video opened: {video_path}")
print(f"[INFO] Resolution: {width}x{height}, FPS: {fps}")

# === Setup output writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

if not out.isOpened():
    print("[ERROR] Could not open VideoWriter!")
    exit()

# === Initialize SORT ===
tracker = Sort()
frame_count = 0

# === Process ===
while True:
    ret, frame = video.read()
    if not ret:
        print("[INFO] End of video or cannot read frame.")
        break

    frame_count += 1
    print(f"[INFO] Processing frame {frame_count}...")

    results = model(frame, verbose=False)  # disable spam
    boxes = []

    for result in results:
        if result.boxes is not None and result.boxes.xyxy is not None:
            xyxys = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.9]*len(xyxys)

            for xyxy, conf in zip(xyxys, confs):
                x1, y1, x2, y2 = map(float, xyxy)
                score = float(conf)
                boxes.append([x1, y1, x2, y2, score])

    if len(boxes) > 0:
        tracks = tracker.update(boxes)
    else:
        tracks = tracker.update([])

    for track in tracks:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)

video.release()
out.release()
print(f"[SUCCESS] Done! Output saved as tracked_output.mp4")
