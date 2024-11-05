#!/usr/bin/python3

import cv2
import numpy as np
import onnxruntime as ort
import time
from picamera2 import Picamera2, Preview

try:
    picam2 = Picamera2()
    picam2.start_preview(Preview.QTGL)

    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()

    model_path = "yolov8n.onnx"
    session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape[2:]
    output_name = session.get_outputs()[0].name
    picam2.set_controls({"AfMode": 1, "AfTrigger": 0})

    time.sleep(1)

    while True:
        try:
            frame = picam2.capture_array()
            
            resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
            img = resized_frame.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            
            result = session.run([output_name], {input_name: img})[0]

            for detection in result:
                x1, y1, x2, y2, conf, label = detection[:6]
                if conf > 0.5:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'class: {int(label)} Conf: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('try yolo', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error during frame processing: {e}")

except Exception as e:
    print(f"Error initializing camera or model: {e}")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
