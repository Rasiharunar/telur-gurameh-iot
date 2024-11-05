import cv2
from picamera2  import Picamera2
from ultralytics import YOLO
from arduino_iot_cloud import ArduinoCloudClient
from RPLCD.i2c import CharLCD

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1,cols=16, rows=2, dotsize=8)
lcd.clear()


picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280,720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode":2, "AfTrigger": 0})

picam2.start()
model = YOLO("best.pt")

client = ArduinoCloudClient(device_id="30523794-6618-41b8-ab29-1e54e3a166c4", secret_key="B@9eAnmG55HFJVeXNXa8JFuLk")


while True:
    frame = picam2.capture_array()
    result = model(frame)
    jumlah_telur = len(result[0].boxes)
    annotated_frame = result[0].plot()
    lcd.clear()
    lcd.write_string(f'Jumlah Telur:\n{jumlah_telur}')

    client.publish_value(property_name="telurIkan", value=jumlah_telur)


    cv2.putText(annotated_frame, f'jumlah telur ikan : {jumlah_telur}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("Hitung Telur Ikan", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break
lcd.clear()
cv2.destroyAllWindows()