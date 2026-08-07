import cv2
print("Testing camera indices...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i} successfully opened and read a frame. Size: {frame.shape}")
        else:
            print(f"Camera {i} opened, but could not read a frame.")
        cap.release()
    else:
        print(f"Camera {i} failed to open.")
