import cv2
import onnxruntime as ort
from picamera2 import Picamera2
import numpy as np

# Inisialisasi kamera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)
picam2.set_controls({"AfMode": 2, "AfTrigger": 0})

picam2.start()

# Load ONNX model
session = ort.InferenceSession("yolov8n.onnx")

# Pemetaan kelas (misalnya, jika indeks 0 mewakili "person")
class_names = ["cell phone","bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant"]

def preprocess(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image, (640, 640))
    input_image = input_image.astype('float32') / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.array(input_image, dtype=np.float32)
    return input_image

def postprocess(image, detections, conf_threshold=0.5):
    h, w, _ = image.shape
    for detection in detections:
        x_center, y_center, width, height, conf = detection[:5]
        class_probs = detection[5:]
        cls = np.argmax(class_probs)
        class_conf = class_probs[cls]

        if class_conf > conf_threshold:  # Hanya tampilkan jika probabilitas lebih besar dari threshold
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # Gambar bounding box dan label pada gambar
            label = f"{class_names[cls]}: {class_conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Loop untuk live video detection
while True:
    # Capture frame dari kamera
    frame = picam2.capture_array()

    # Pastikan frame memiliki 3 channel
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess frame
    input_image = preprocess(frame)
    input_image = np.expand_dims(input_image, axis=0)

    # Inferensi model
    outputs = session.run(None, {session.get_inputs()[0].name: input_image})
    detections = outputs[0][0]  # Assuming first batch
    detections = detections.reshape(-1, 6)  # Adjust based on actual format
    frame = postprocess(frame, detections)

    # Tampilkan frame hasil deteksi
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cv2.destroyAllWindows()
picam2.stop()
