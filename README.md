# FaceLocking

Face detection, enrollment, ArcFace embedding, recognition, and selected-face tracking with OpenCV, MediaPipe, and ONNX Runtime.

## Windows setup

Open PowerShell in this repository and install the pinned Python packages:

```powershell
python -m pip install -r requirements.txt
```

The ArcFace ONNX model is not stored in Git because it is large. Put the model at `models/embedder_arcface.onnx`. The MediaPipe task model and OpenCV Haar cascade are included under `data/`.

Check that Python can import the main dependencies:

```powershell
python -c "import cv2, mediapipe, numpy, onnxruntime; print('Dependencies OK')"
```

## Test the project in stages

Run commands from the repository root. Press `q` to close camera previews.

### 1. Check camera access

```powershell
python -m src.camera
```

These basic checks open camera index `0`. The lock-and-track command below supports selecting a camera with `--camera` (often `0` for built-in and `1` for an external USB camera).

### 2. Check face detection

```powershell
python -m src.detect
```

The preview should draw a box around a face. If it cannot open the camera, close other apps using the camera and check Windows camera permissions.

### 3. Enroll your identity

```powershell
python -m src.enroll --camera 0
```

Enter your identity name in the PowerShell prompt. Keep your face well lit and centered. Capture at least 15 varied, sharp samples, looking straight ahead and slightly changing your pose. Use **SPACE** for one capture or **a** to toggle automatic capture, then press **s** to save. Press **q** to quit. The tool writes the local database to `data/db/face_db.npz` and `data/db/face_db.json`; aligned captures are saved under `data/enroll/<name>/`.

Those database and enrollment files are intentionally ignored by Git because they contain biometric data. Enroll again on a new computer, or transfer them privately if you need the same identity database.

### 4. Inspect embedding output

```powershell
python -m src.embed
```

This opens the live alignment and embedding visualization. Press **P** to print embedding statistics and **Q** to quit. Embeddings are also generated as part of enrollment and recognition; this step is a visualization/debug check.

### 5. Tune recognition threshold (optional)

After enrolling one or more identities, run:

```powershell
python -m src.evaluate
```

This evaluates saved enrollment crops and suggests a recognition threshold. Use several clean samples per identity for meaningful results.

### 6. Test multi-face recognition

```powershell
python -m src.recognize
```

The enrolled identity should be labeled when recognized; people without a matching template should be rejected as unknown. Test with more than one person in view and watch for false matches.

### 7. Test identity lock, expressions, and nose position

Use the exact identity name used during enrollment:

```powershell
python -m src.face_tracking --target "Nora" --camera 0 --width 1280 --height 720
```

Use the same `--camera` option when enrolling, for example `python -m src.enroll --camera 1 --width 1280 --height 720`. For an external HD camera, try index `1`; if it is a different device index, try `0`, `1`, or `2`. The tracking overlay should lock the selected identity, mark other detected faces as ignored, display blink and expression cues, and show the nose tip's direction and pixel distance from frame center. Change `--lost-timeout 2.0` to set how long it keeps the target lock during a brief disappearance.

The expression overlay asks you to hold a neutral face briefly while it learns your baseline. Then try a smile, a clear frown, a sad expression with raised inner brows and downturned mouth corners, and a tight grimace one at a time. Hold each for a second so the smoothed signal can respond. Labels combine MediaPipe blendshape scores with face-landmark geometry; they are approximate cues and depend on lighting, camera angle, and face visibility. The reported nose distance is in image pixels, not physical units. The camera driver may ignore requested resolutions; verify the actual preview size before presenting an HD-camera test.

## Privacy and repository contents

The Git repository contains source code and shared model assets only. Personal face crops, identity databases, Python caches, and the large ArcFace ONNX model are excluded by `.gitignore`. Do not commit biometric files or model files unless you have reviewed their permissions and distribution terms.
