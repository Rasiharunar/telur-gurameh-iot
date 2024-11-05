import onnxruntime as ort

# Muat model ONNX
session = ort.InferenceSession("best.onnx")

# Mendapatkan informasi tentang input
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

print(f"Input name: {input_name}")
print(f"Input shape: {input_shape}")
print(f"Input type: {input_type}")

# Mendapatkan informasi tentang output
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape
output_type = session.get_outputs()[0].type

print(f"Output name: {output_name}")
print(f"Output shape: {output_shape}")
print(f"Output type: {output_type}")
