import cv2
from ultralytics import YOLO

IMG_PATH = "content/test2.jpg"               # <- giriş görseli
OUT_PATH = "content/test2_pred.jpg"          # <- çıktı kaydı
MODEL_PATH = "models/best.pt"                # <- eğittiğin model

CONF_THRES = 0.25
IOU_THRES  = 0.50

# Model
model = YOLO(MODEL_PATH)

# Görseli oku (BGR)
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı: {IMG_PATH}")

# Tahmin
results = model(img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)

# Çizimli çıktı al
annotated = results[0].plot()  # Ultralytics'in kendi çizimi

# Kaydet
cv2.imwrite(OUT_PATH, annotated)
print(f"Kaydedildi: {OUT_PATH}")
