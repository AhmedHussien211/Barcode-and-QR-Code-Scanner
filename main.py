import cv2
from pyzbar.pyzbar import decode

frame = cv2.imread("MultipleQR_Bar_code.PNG", cv2.IMREAD_COLOR)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    if len(approx) == 4:  
        x,y,w,h = cv2.boundingRect(cnt)
        roi = frame[y:y+h, x:x+w]

decoded_objects = decode(frame)
for obj in decoded_objects:
    points = obj.polygon
    data = obj.data.decode("utf-8")
    print("Decoded:", data)
    cv2.putText(frame, data, (obj.rect.left, obj.rect.top-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Barcode/QR Scanner", frame)
cv2.waitKey(0)

