import cv2
import signal
import sys
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from RPLCD.i2c import CharLCD
from flask import Flask, Response, render_template
import time
app = Flask(__name__)

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (480, 360)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
picam2.set_controls({"FrameRate": 15.0})  # Set to 15 FPS

picam2.start()
lcd.write_string(f'Kamera Aktif')

model = YOLO("best.pt")

def update_lcd(jumlah_telur):
    lcd.clear()
    lcd.write_string(f'Jumlah Telur:\n{jumlah_telur}')

def shutdown_handler(signum, frame):
    print("Shutdown signal received. Cleaning up...")
    picam2.stop()  
    lcd.clear()  
    cv2.destroyAllWindows()  
    sys.exit(0) 

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

def detection_loop():
    while True:
        frame = picam2.capture_array()
        result = model.predict(frame, max_det=2000)
        jumlah_telur = len(result[0].boxes)
        
        # Update LCD with the number of detected eggs
        update_lcd(jumlah_telur)

        # Optional: You can add a delay here if needed to reduce CPU usage
        time.sleep(1)  # Adjust the sleep time as necessary

# Start the detection loop in a separate thread
threading.Thread(target=detection_loop, daemon=True).start()

def generate_frames():
    while True:
        frame = picam2.capture_array()
        result = model.predict(frame, max_det=2000)
        annotated_frame = result[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('template/dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)