"""Record tracker fixtures (see td_tracking_recorder.py). POST to the bridge's /run:

	curl -X POST --data-binary @tests/td_record_tracking.py http://127.0.0.1:3031/run

A session's 'motion' (0-1) moves, scales and turns a still photo the same way every run (faces:
there's no face clip). Returns at once; build/tracking/recorded.json appears when every session
is done. Then:

	python tests/tracking_replay.py build/tracking/<session>.jsonl.gz

ONLY limits the run to these session names (empty = all).
"""
import importlib
import sys
from pathlib import Path

ONLY = ['holistic_gpu_chestup', 'pose_chestup', 'yolopose_chestup', 'movenet_chestup']
MEDIA = (Path(project.folder) / 'media').as_posix()
PHOTOS = (Path(project.folder) / 'docs' / 'images' / 'sources').as_posix()

SESSIONS = [
	# Skeleton modes (phase 1).
	{'name': 'holistic_gpu_dancer', 'wrapper': 'MediaPipeHolisticGPU', 'source': f'{MEDIA}/dancer.mp4', 'frames': 300},
	{'name': 'holistic_gpu_noface_dancer', 'wrapper': 'MediaPipeHolisticGPU', 'instance': 'MediaPipeHolisticGPU_noface',
		'source': f'{MEDIA}/dancer.mp4', 'frames': 150, 'pars': {'Holisticface': False}},
	{'name': 'holistic_cpu_dancer', 'wrapper': 'MediaPipeHolistic', 'source': f'{MEDIA}/dancer.mp4', 'frames': 300},
	{'name': 'pose_twins', 'wrapper': 'MediaPipePose', 'source': f'{MEDIA}/les-twins-wod-2014.mp4', 'frames': 300},
	{'name': 'hands_clip', 'wrapper': 'MediaPipeHands', 'source': f'{MEDIA}/hands.mp4', 'frames': 300},
	{'name': 'face_moving', 'wrapper': 'MediaPipeFace', 'source': f'{PHOTOS}/face.jpg', 'frames': 300, 'motion': 1.0},
	{'name': 'face_photo', 'wrapper': 'MediaPipeFace', 'source': f'{PHOTOS}/face.jpg', 'frames': 40},
	{'name': 'facedet_moving', 'wrapper': 'MediaPipeFaceDetector', 'source': f'{PHOTOS}/face.jpg', 'frames': 300, 'motion': 1.0},
	{'name': 'yolopose_walking', 'wrapper': 'Yolo26Pose', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'movenet_walking', 'wrapper': 'MoveNet', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'rtmw_dancer', 'wrapper': 'RTMW3D', 'source': f'{MEDIA}/dancer.mp4', 'frames': 300},
	# Chest-up framing (a desk webcam): the dancer's close-up, zoomed so the legs are out of the image
	# and the hips at its bottom edge (clip frames 270-335; before and after, the knees are at the
	# edge). The models still report legs; the tracker's joint gate (tests/test_joint_gate.py) and
	# the puppet are judged on these.
	*[{'name': f'{name}_chestup', 'wrapper': wrapper, 'source': f'{MEDIA}/dancer.mp4', 'start': 270, 'frames': 66,
		'view': (1.8, 0.54, 0.55)} for name, wrapper in (('holistic_gpu', 'MediaPipeHolisticGPU'), ('pose', 'MediaPipePose'),
		('yolopose', 'Yolo26Pose'), ('movenet', 'MoveNet'))],
	# Box and face modes (phase 3).
	{'name': 'detect_walking', 'wrapper': 'Yolo26Detect', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'segment_walking', 'wrapper': 'Yolo26Seg', 'source': f'{MEDIA}/people-walking.mp4', 'frames': 300},
	{'name': 'yunet_walking', 'wrapper': 'YuNet', 'source': f'{MEDIA}/people-walking-forward.mp4', 'frames': 300},
	{'name': 'yunet_moving', 'wrapper': 'YuNet', 'source': f'{PHOTOS}/face.jpg', 'frames': 300, 'motion': 1.0},
	{'name': 'yunet_photo', 'wrapper': 'YuNet', 'source': f'{PHOTOS}/face.jpg', 'frames': 40},
	{'name': 'yunet_solvepnp_moving', 'wrapper': 'YuNet', 'instance': 'YuNet_solvepnp', 'source': f'{PHOTOS}/face.jpg',
		'frames': 300, 'motion': 1.0, 'pars': {'Trackposemethod': 'solvepnp'}},
]

tests_dir = (Path(project.folder) / 'tests').as_posix()
if tests_dir not in sys.path:
	sys.path.insert(0, tests_dir)
import td_tracking_recorder
importlib.reload(td_tracking_recorder)

sessions = [s for s in SESSIONS if not ONLY or s['name'] in ONLY]
td_tracking_recorder.start(sessions)
print(f'[TrackingRecorder] recording {len(sessions)} sessions: {[s["name"] for s in sessions]}')
