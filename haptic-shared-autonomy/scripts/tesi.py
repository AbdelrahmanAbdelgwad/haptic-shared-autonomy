import cv2

cap = cv2.VideoCapture('192.168.1.1')

if not cap.isOpened():
    print("No connect")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()