# Installs the official MediaPipe Tasks runtime + models for MediaPipeTOP:
#   plugins/libmediapipe.dll            <- from the mediapipe==1.2.2 win_amd64 wheel (mediapipe/tasks/c/)
#   models/mediapipe/*.task|.tflite <- Google's model bundles (storage.googleapis.com/mediapipe-models)
# The C headers in third_party/mediapipe are from the same tag (v1.2.2); keep the versions in sync.
param(
	[string]$Version = '1.2.2',
	[switch]$Heavy		# also fetch the slow ones: pose_landmarker_heavy (~40 ms/frame on CPU) and the
						# 6-class selfie_multiclass_256x256 segmenter (~156 ms/frame)
)
$ErrorActionPreference = 'Stop'
$repo = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$plugins = Join-Path $repo 'plugins'
$models = Join-Path $repo 'models\mediapipe'
New-Item -ItemType Directory -Force $plugins, $models | Out-Null

# --- runtime DLL (the wheel is a zip; only the one DLL is extracted, nothing is installed)
$tmp = Join-Path ([IO.Path]::GetTempPath()) "mediapipe_wheel_$Version"
New-Item -ItemType Directory -Force $tmp | Out-Null
$wheel = Get-ChildItem $tmp -Filter "mediapipe-$Version-*win_amd64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $wheel) {
	python -m pip download "mediapipe==$Version" --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.11 -d $tmp -q
	$wheel = Get-ChildItem $tmp -Filter "mediapipe-$Version-*win_amd64.whl" | Select-Object -First 1
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($wheel.FullName)
try {
	$entry = $zip.Entries | Where-Object { $_.FullName -eq 'mediapipe/tasks/c/libmediapipe.dll' }
	if (-not $entry) { throw "libmediapipe.dll not found in $($wheel.Name)" }
	$dst = Join-Path $plugins 'libmediapipe.dll'
	try {
		[IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dst, $true)
		Write-Host "libmediapipe.dll $Version -> $dst"
	} catch {
		Write-Warning "Could not write $dst (TouchDesigner has it loaded?) -- close TD and re-run."
	}
} finally { $zip.Dispose() }

# --- model bundles
$base = 'https://storage.googleapis.com/mediapipe-models'
$tasks = @(
	'face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
	'hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
	'gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task',
	'pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
	'pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
	'holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
	'face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
	'image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
	'image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite',
	'image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite'
)
if ($Heavy) {
	$tasks += 'pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
	$tasks += 'image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
}
foreach ($t in $tasks) {
	$out = Join-Path $models (Split-Path $t -Leaf)
	if (Test-Path $out) { Write-Host "have $(Split-Path $t -Leaf)"; continue }
	Invoke-WebRequest "$base/$t" -OutFile $out
	Write-Host "fetched $(Split-Path $t -Leaf)"
}
