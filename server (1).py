"""
EGSL — FastAPI Server for Koyeb
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import json, os
from collections import deque, Counter

app = FastAPI(title="EGSL API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  FRANCO → ARABIC
# ─────────────────────────────────────────────
FRANCO_TO_ARABIC = {
    '3amel eh':'عامل إيه؟','3ieb':'عيب','3rabia':'عربية','3waz':'عوز',
    'akhoia':'أخويا','alam':'ألم','bhab':'بحب','bkrh':'بكره',
    'bsor3a':'بسرعة','delwaty':'دلوقتي','docktor':'دكتور','efrda':'افردة',
    'ehtrem nfsk':'احترم نفسك','elnhrda':'النهارده',
    'erny wra dhrk':'ورّيني ورا ضهرك','fatan':'فطان','fe sark':'في سرك',
    'flos':'فلوس','fokk many':'فك مني','gomma':'جمعة','harmy':'حرامي',
    'hba hba':'هبة هبة','helw awy':'حلو أوي','hta hta':'حتة حتة',
    'kdab':'كداب','khaf':'خاف','khaly balk':'خلي بالك','khatbty':'خطبتي',
    'kolya':'كلية','ktab':'كتاب','mafish':'مفيش','makar':'مكر',
    'manzel':'منزل','mash mashy':'مش ماشي','mashy':'ماشي','meraty':'مراتي',
    'mohands':'مهندس','moshwash':'مشواش','msh 3waz':'مش عوز','oda':'أوضة',
    'sabhl elkheer':'صبح الخير','sabhlala':'صبحلالة','sadak':'صدق',
    'sahby':'صاحبي','sbak':'صباك','shar3':'شارع','shbak':'شباك',
    't3ban':'تعبان','tmam':'تمام','whashny':'وحشني','whda whda':'وحدة وحدة',
    'y3takr':'يعتذر','yabky':'يبكي','yadhk':'يضحك','yallhwy':'يلا هوي',
    'yshtm':'يشتم',
}

SEQUENCE_LEN = 42
FEATURES     = 258

# ─────────────────────────────────────────────
#  LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
model    = None
cls_f    = []
cls_a    = []
norm_mean = None
norm_std  = None

@app.on_event("startup")
async def startup():
    global model, cls_f, cls_a, norm_mean, norm_std

    # Model
    model = tf.keras.models.load_model("egsl_model_v2.h5")
    print(f"✅ Model loaded — input: {model.input_shape}")

    # Label map
    with open("label_map.json", "r", encoding="utf-8") as f:
        lm = json.load(f)

    if "franco" in lm:
        n = len(lm["franco"])
        cls_f = [""]*n; cls_a = [""]*n
        for name, idx in lm["franco"].items():
            cls_f[idx] = name
            cls_a[idx] = lm.get("arabic", {}).get(
                name, FRANCO_TO_ARABIC.get(name, name))
    else:
        n = len(lm)
        cls_f = [""]*n; cls_a = [""]*n
        for name, idx in lm.items():
            cls_f[idx] = name
            cls_a[idx] = FRANCO_TO_ARABIC.get(name, name)

    norm_mean = np.load("norm_mean.npy")
    norm_std  = np.load("norm_std.npy")
    print(f"✅ {n} classes loaded")

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[lm.x, lm.y, lm.z]
                     for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def process_image_to_keypoints(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        results = holistic.process(rgb)
    return extract_keypoints(results)

# ─────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status"  : "✅ EGSL API running",
        "classes" : len(cls_f),
        "model"   : "BiLSTM",
        "usage"   : "POST /predict with frames[]"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/classes")
def get_classes():
    return {
        "classes": [
            {"id": i, "franco": cls_f[i], "arabic": cls_a[i]}
            for i in range(len(cls_f))
        ]
    }

@app.post("/predict")
async def predict(frames: list[UploadFile] = File(...)):
    """
    استقبل list من الصور (frames) وارجع النتيجة.
    Flutter بيبعت 20-42 صورة → السيرفر يحللهم ويرجع الكلمة.
    """
    if model is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=503)

    if len(frames) < 5:
        return JSONResponse(
            {"error": f"محتاج على الأقل 5 frames، بعت {len(frames)}"},
            status_code=400
        )

    # استخرج الـ keypoints من كل frame
    sequence = []
    for frame in frames:
        img_bytes = await frame.read()
        kp = process_image_to_keypoints(img_bytes)
        sequence.append(kp)

    # Pad أو Trim للـ SEQUENCE_LEN
    while len(sequence) < SEQUENCE_LEN:
        sequence.append(sequence[-1])
    sequence = sequence[:SEQUENCE_LEN]

    # Normalize
    arr  = np.array(sequence, dtype=np.float32)
    norm = (arr - norm_mean) / (norm_std + 1e-8)

    # Predict
    pred     = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
    top3_idx = np.argsort(pred)[::-1][:3]
    top_idx  = int(top3_idx[0])
    top_conf = float(pred[top_idx])

    return {
        "prediction": {
            "arabic"    : cls_a[top_idx],
            "franco"    : cls_f[top_idx],
            "confidence": round(top_conf * 100, 1),
        },
        "top3": [
            {
                "arabic"    : cls_a[i],
                "franco"    : cls_f[i],
                "confidence": round(float(pred[i]) * 100, 1),
            }
            for i in top3_idx
        ],
        "frames_received": len(frames),
    }
