from sys import argv
from ipywidgets import Video
from pathlib import Path
import cv2

script, filename = argv
out_dir = Path('../CPGs')

if len(argv) > 2:
    print(argv[2])
    out_dir = Path(argv[2])

cap = cv2.VideoCapture(str(out_dir/filename))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    k = cv2.waitKey(24)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


