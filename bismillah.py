import cv2
import signal
import sys
import threading
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from RPLCD.i2c import CharLCD
import socket

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (720, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
picam2.set_controls({"FrameRate": 15.0})  # Set to 15 FPS

picam2.start()
lcd.write_string(f'Kamera Aktif')

model = YOLO("new25okt.pt")
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


def trigger_focus():
    global picam2
    picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
    time.sleep(3)

def draw_button(frame, text, x, y, w, h, color=(0,255,0),  thickness=2):
    cv2.rectangle(frame, (x, y), (x + w, y + h),color,thickness)
    cv2.putText(frame, text, (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def handle_mouse(event, x, y, flags, param):
    global button_x, button_y, button_width, button_height
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            trigger_focus()

button_x = 10
button_y = 10
button_width = 150
button_height = 50
cv2.namedWindow("Hitung Telur Ikan")
cv2.setMouseCallback("Hitung Telur Ikan", handle_mouse)


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
    while True:
        frame = picam2.capture_array()
          
        result = model.predict(frame, max_det=2000)
        jumlah_telur = len(result[0].boxes)
        annotated_frame = result[0].plot()
        for box in result[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        draw_button(frame, "Autofokus", button_x, button_y, button_width, button_height)

        jumlah_telur_window = f'Deteksi Telur Ikan - Jumlah Telur : {jumlah_telur}'
       
        threading.Thread(target=update_lcd, args=(jumlah_telur,)).start()
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.setWindowTitle("Hitung Telur Ikan", jumlah_telur_window)
        
        cv2.imshow("Hitung Telur Ikan", frame)
        

        if cv2.waitKey(1) == ord("q"):
            lcd.clear()
            time.sleep(3)
            break

except Exception as e:
    print(f"Error: {e}")
    shutdown_handler(None, None)

shutdown_handler(None, None)