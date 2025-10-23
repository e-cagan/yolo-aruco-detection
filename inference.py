"""
Training Notebook Link: https://colab.research.google.com/drive/1SJtGx56i-CU3tZa3kLhST6KzW0nsR4gH#scrollTo=xeRSPI0_ylKU
"""

import os
os.environ["QT_LOGGING_RULES"] = "*.debug=false"  # Qt thread uyarılarını susturur (bilgi amaçlı)

import cv2
import time
import numpy as np
from ultralytics import YOLO

# ========= Ayarlar =========
MODEL_PATH = 'models/best.pt'
CONF_THRES = 0.40      # FP azalsın
IOU_THRES  = 0.45
MAX_ROI    = 10

# Sadece beklediğin sözlük (tek tip ise bunu bırak)
DICT_ID = cv2.aruco.DICT_4X4_50  # ihtiyacına göre değiştir

# Opsiyonel kamera matrisi (pose istiyorsan)
CAMERA_MATRIX = None
DIST_COEFFS   = None
MARKER_SIZE_M = None

# ========= ArUco Detector (versiyon güvenli) =========
def make_aruco_detector(dict_id):
    adict = cv2.aruco.getPredefinedDictionary(dict_id)
    # Parametreler
    params = (cv2.aruco.DetectorParameters_create()
              if hasattr(cv2.aruco, "DetectorParameters_create")
              else cv2.aruco.DetectorParameters())
    
    # Biraz sıkılaştır (FP azaltır)
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
    params.minMarkerPerimeterRate = 0.02   # çok küçükleri ele
    params.maxMarkerPerimeterRate = 4.0
    params.minCornerDistanceRate = 0.05

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(adict, params)
        def detect(img):
            return detector.detectMarkers(img)
    else:
        def detect(img):
            return cv2.aruco.detectMarkers(img, adict, parameters=params)
    return detect

detect_aruco = make_aruco_detector(DICT_ID)

# ========= Model & Capture =========
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

t_last = time.time()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Kamera açılamadı.")
        break

    # YOLO inference
    res = model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
    vis = frame.copy()

    kept = 0
    for box in res.boxes:
        if kept >= MAX_ROI:
            break
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(vis.shape[1]-1, x2); y2 = min(vis.shape[0]-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        roi = vis[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # ArUco decode: gri + hafif iyileştirme (daha stabil)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 0)

        corners, ids, _ = detect_aruco(gray)

        # Sıkı filtre: decode yoksa KUTU ÇİZME
        if ids is None or len(ids) == 0:
            continue

        kept += 1
        # Çizimler
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.aruco.drawDetectedMarkers(roi, corners, ids)
        cv2.putText(roi, f"ID:{int(ids[0][0])}", (5,18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,220,255), 2, cv2.LINE_AA)

        # Pose (isteğe bağlı)
        if CAMERA_MATRIX is not None and DIST_COEFFS is not None and MARKER_SIZE_M:
            try:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
                )
                cv2.aruco.drawAxis(roi, CAMERA_MATRIX, DIST_COEFFS,
                                   rvecs[0], tvecs[0], MARKER_SIZE_M * 0.5)
            except Exception:
                pass

    # FPS
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/(now - t_last))
    t_last = now
    cv2.putText(vis, f"FPS:{fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2, cv2.LINE_AA)

    cv2.imshow("YOLO + ArUco (filtered)", vis)
    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
