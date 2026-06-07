import cv2
import os

name = input("Enter Person Name: ")

path = f"dataset/{name}"
os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0)

count = 0

while count < 100:
    ret, frame = cam.read()

    if not ret:
        break

    cv2.imshow("Capture Images", frame)

    cv2.imwrite(f"{path}/{count}.jpg", frame)

    count += 1

    if cv2.waitKey(100) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

print("Images Captured Successfully")