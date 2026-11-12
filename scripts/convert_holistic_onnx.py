"""Convert MediaPipe Holistic's landmark models (models/mediapipe/holistic_landmarker.task) to ONNX
for OnnxTOP: tflite -> tf2onnx -> NCHW float input named `image`, unused outputs dropped. Output in
models/onnx/mediapipe_holistic/. Run outside TD, in a throwaway Python 3.11 venv:

	uv venv -p 3.11 conv311
	uv pip install -p conv311 "tensorflow-cpu==2.15.1" "tf2onnx==1.16.1" "onnx==1.16.2" "numpy<2" onnxruntime
	conv311\\Scripts\\python scripts\\convert_holistic_onnx.py

Each converted model is checked against the TFLite original on random input. The pose detector
is skipped: it uses sparse DENSIFY weights tf2onnx can't convert (YOLO26 finds people instead).
"""
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = os.path.join(ROOT, 'models', 'mediapipe', 'holistic_landmarker.task')
OUT = os.path.join(ROOT, 'models', 'onnx', 'mediapipe_holistic')
MODELS = ['pose_landmarks_detector', 'hand_landmarks_detector', 'hand_roi_refinement', 'face_landmarks_detector', 'face_detector']
DROP = {'pose_landmarks_detector': {'Identity_2'}}	# the 256x256 segmentation mask


def to_nchw(src, dst, drop):
	m = onnx.load(src)
	g = m.graph
	inp = g.input[0]
	_, h, w, c = [d.dim_value for d in inp.type.tensor_type.shape.dim]
	nhwc = inp.name + '_nhwc'
	for node in g.node:
		node.input[:] = [nhwc if i == inp.name else i for i in node.input]
	g.input.remove(inp)
	g.input.insert(0, helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, c, h, w]))
	g.node.insert(0, helper.make_node('Transpose', ['image'], [nhwc], perm=[0, 2, 3, 1], name='nchw_to_nhwc'))
	for o in [o for o in g.output if o.name in drop]:
		g.output.remove(o)
	m = onnx.shape_inference.infer_shapes(m)
	onnx.checker.check_model(m)
	onnx.save(m, dst)
	return h, w, c


def check(tflite_path, onnx_path, h, w, c):
	x = np.random.default_rng(0).random((1, h, w, c), dtype=np.float32)
	it = tf.lite.Interpreter(model_path=tflite_path)
	it.allocate_tensors()
	it.set_tensor(it.get_input_details()[0]['index'], x)
	it.invoke()
	ref = {o['name']: it.get_tensor(o['index']) for o in it.get_output_details()}
	sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
	got = sess.run(None, {'image': x.transpose(0, 3, 1, 2).copy()})
	return max(float(np.abs(v - ref[o.name]).max()) for o, v in zip(sess.get_outputs(), got))


def main():
	os.makedirs(OUT, exist_ok=True)
	with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(TASK) as task:
		for name in MODELS:
			tfl = os.path.join(tmp, name + '.tflite')
			with open(tfl, 'wb') as f:
				f.write(task.read(name + '.tflite'))
			raw = os.path.join(tmp, name + '.onnx')
			subprocess.run([sys.executable, '-m', 'tf2onnx.convert', '--tflite', tfl, '--output', raw, '--opset', '17'],
				check=True, capture_output=True)
			dst = os.path.join(OUT, name + '.onnx')
			h, w, c = to_nchw(raw, dst, DROP.get(name, set()))
			print(f'[convert_holistic_onnx] {name} {w}x{h}: max diff vs tflite {check(tfl, dst, h, w, c):.2e}')


if __name__ == '__main__':
	main()
