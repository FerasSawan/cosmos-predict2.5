"""Resize a single video. Run as a standalone process."""
import sys
import cv2

input_path = sys.argv[1]
output_path = sys.argv[2]
width = int(sys.argv[3])
height = int(sys.argv[4])

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print(f"ERROR: Cannot open {input_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    count += 1
    if count % 5000 == 0:
        print(f"  {count}/{total} frames", flush=True)

cap.release()
out.release()
print(f"Done: {count} frames written")
