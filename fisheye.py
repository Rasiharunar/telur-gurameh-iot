import cv2
import signal
import sys
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
from RPLCD.i2c import CharLCD
import socket

# Inisialisasi LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()

# Inisialisasi kamera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (480, 360)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
picam2.set_controls({"FrameRate": 15.0})  # Set to 15 FPS

picam2.start()
lcd.write_string('Kamera Aktif')

# Load model YOLO
model = YOLO("best.pt")

def get_ip_address():
    """Get the Raspberry Pi's IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Menghubungkan ke alamat DNS Google
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "0.0.0.0"
    finally:
        s.close()
    return ip_address

def update_lcd(jumlah_telur):
    ip_address = get_ip_address()
    lcd.clear()
    lcd.cursor_pos = (0, 0)
    lcd.write_string(f'Jumlah Telur:{jumlah_telur}')
    lcd.cursor_pos = (1, 0)
    lcd.write_string(f'{ip_address}')

def shutdown_handler(signum, frame):
    print("Shutdown signal received. Cleaning up...")
    picam2.stop()  
    lcd.clear()  
    cv2.destroyAllWindows()  
    sys.exit(0) 

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

try:
    # Tampilkan IP di LCD
    
    while True:
        frame = picam2.capture_array()
        result = model.predict(frame, max_det=2000)
        jumlah_telur = len(result[0].boxes)
        annotated_frame = result[0].plot()
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        jumlah_telur_window = f'Deteksi Telur Ikan - Jumlah Telur : {jumlah_telur}'
        
        threading.Thread(target=update_lcd, args=(jumlah_telur,)).start()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow("Hitung Telur Ikan", frame)
        cv2.setWindowTitle("Hitung Telur Ikan", jumlah_telur_window)
        if cv2.waitKey(1) == ord("q"):
            lcd.clear()
            break

except Exception as e:
    print(f"Error: {e}")
    shutdown_handler(None, None)

shutdown_handler(None, None)