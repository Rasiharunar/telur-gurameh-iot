import cv2
import onnxruntime as ort
from picamera2 import Picamera2
from RPLCD.i2c import CharLCD
import numpy as np

lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)
lcd.clear()

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.set_controls({"AfMode": 2, "AfTrigger": 0})

picam2.start()

# Load the ONNX model
session = ort.InferenceSession("best.onnx")

# Get input and output names for the ONNX model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

while True:
    frame = picam2.capture_array()

    # Resize the image to the required input size of the model
    resized_frame = cv2.resize(frame, (800, 800))  # Resize image to [800, 800]
    input_image = resized_frame.transpose(2, 0, 1)  # Change data layout from HWC to CHW
    input_image = input_image[np.newaxis, :, :, :].astype(np.float32)  # Add batch dimension and convert to float32

    # Run the model
    result = session.run([output_name], {input_name: input_image})[0]

    # Post-process the result
    jumlah_telur = result.shape[2]  # Assuming this shape represents the number of detected objects
    annotated_frame = resized_frame.copy()
    cv2.putText(annotated_frame, f'jumlah telur ikan : {jumlah_telur}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Update the LCD display
    lcd.clear()
    lcd.write_string(f'Jumlah Telur:\n{jumlah_telur}')

    # Display the annotated frame
    cv2.imshow("Hitung Telur Ikan", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break
lcd.clear()
cv2.destroyAllWindows()
