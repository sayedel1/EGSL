"""
EGSL — Egyptian Sign Language Recognition
Standalone OpenCV App — زي الصورة بالظبط
تشغيل: python run.py
"""

import cv2
import mediapipe as mp
import numpy as np
import json, os, time
import tensorflow as tf
from collections import deque, Counter

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE       = r"C:\Users\SAYED\Desktop\EGSL\test model"
MODEL_PATH = os.path.join(BASE, "EGSL_Model",     "egsl_bilstm_best.keras")
LABEL_PATH = os.path.join(BASE, "EGSL_Processed", "label_map.json")
MEAN_PATH  = os.path.join(BASE, "EGSL_Processed", "norm_mean.npy")
STD_PATH   = os.path.join(BASE, "EGSL_Processed", "norm_std.npy")

SEQUENCE_LEN = 42
CONF_THR     = 0.60

# ─────────────────────────────────────────────
#  COLORS (BGR)
# ─────────────────────────────────────────────
GOLD   = (76, 168, 201)
GREEN  = (113, 237, 46)
RED    = (87, 71, 255)
YELLOW = (0, 215, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (16, 16, 24)

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

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
print("⏳ جاري تحميل الموديل...")
model    = tf.keras.models.load_model(MODEL_PATH)
mean     = np.load(MEAN_PATH)
std      = np.load(STD_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    lm = json.load(f)

if "franco" in lm:
    n = len(lm["franco"])
    cls_f = [""]*n; cls_a = [""]*n
    for name, idx in lm["franco"].items():
        cls_f[idx] = name
        cls_a[idx] = lm.get("arabic", {}).get(name, FRANCO_TO_ARABIC.get(name, name))
else:
    n = len(lm)
    cls_f = [""]*n; cls_a = [""]*n
    for name, idx in lm.items():
        cls_f[idx] = name
        cls_a[idx] = FRANCO_TO_ARABIC.get(name, name)

print(f"✅ الموديل جاهز — {n} كلمة")

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    pose = np.array([[lm.x,lm.y,lm.z,lm.visibility]
                     for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[lm.x,lm.y,lm.z]
                     for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[lm.x,lm.y,lm.z]
                     for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def draw_landmarks(image, results):
    # Pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=GOLD, thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(150, 210, 240), thickness=1))
    # Hands
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            mp_drawing.draw_landmarks(
                image, hand, mp_holistic.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style())
    return image


def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2, radius=10):
    """مستطيل بزوايا دائرية."""
    cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
    cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0, 90,  color, thickness)


def draw_filled_rect(img, x1, y1, x2, y2, color, alpha=0.6):
    """مستطيل شفاف."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)


def put_text_center(img, text, cx, cy, font_scale=0.9,
                    color=WHITE, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(img, text, (cx - tw//2, cy + th//2),
                font, font_scale, color, thickness, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, value, max_val, color):
    """Progress bar."""
    cv2.rectangle(img, (x,y), (x+w, y+h), (50,50,50), -1)
    filled = int(w * min(value/max_val, 1.0))
    if filled > 0:
        cv2.rectangle(img, (x,y), (x+filled, y+h), color, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), (100,100,100), 1)


def draw_hand_box(img, results, pred_text, conf, color):
    """ارسم box حوالين الإيد زي الصورة المرجعية."""
    h, w = img.shape[:2]
    pts  = []

    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm_ in hand.landmark:
                pts.append((int(lm_.x*w), int(lm_.y*h)))

    if not pts:
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 20
    x1 = max(0, min(xs)-pad)
    y1 = max(0, min(ys)-pad)
    x2 = min(w, max(xs)+pad)
    y2 = min(h, max(ys)+pad)

    # Box حوالين الإيد
    draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=2)

    # Label فوق الـ box زي الصورة
    label = f"{pred_text}: {conf:.0f}%"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    (lw, lh), _ = cv2.getTextSize(label, font, 0.65, 2)
    # خلفية اللabel
    cv2.rectangle(img, (x1, y1-lh-8), (x1+lw+10, y1), color, -1)
    cv2.putText(img, label, (x1+5, y1-5),
                font, 0.65, BLACK, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  OVERLAY UI
# ─────────────────────────────────────────────
def draw_ui(img, sequence_len, cur_len, cur_arabic, cur_franco,
            cur_conf, history, total, fps):

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # ── TOP BAR ──
    draw_filled_rect(img, 0, 0, w, 55, DARK, alpha=0.75)
    title = "EGSL - Egyptian Sign Language"
    cv2.putText(img, title, (12, 35), font, 0.75, (*GOLD[::-1],), 2, cv2.LINE_AA)
    fps_txt = f"FPS: {fps:.1f}"
    (fw,_),_ = cv2.getTextSize(fps_txt, font, 0.6, 1)
    cv2.putText(img, fps_txt, (w-fw-12, 35), font, 0.6, GREEN, 1, cv2.LINE_AA)

    # ── LEFT PANEL — Result ──
    panel_w = 260
    draw_filled_rect(img, 0, 55, panel_w, h, DARK, alpha=0.70)
    cv2.line(img, (panel_w, 55), (panel_w, h), (*GOLD[::-1],), 1)

    # Prediction label
    cy = 100
    cv2.putText(img, "PREDICTION", (10, cy), font, 0.5,
                (*GOLD[::-1],), 1, cv2.LINE_AA)
    cy += 10
    cv2.line(img, (10, cy), (panel_w-10, cy), (*GOLD[::-1],), 1)

    # Arabic word (كبير)
    cy += 45
    if cur_arabic and cur_arabic != "—":
        # نقسم الكلمة لو طويلة
        words = cur_arabic.split()
        for i, word in enumerate(words[:2]):
            cv2.putText(img, word, (15, cy + i*35),
                        font, 0.9, WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "---", (15, cy), font, 0.9, (100,100,100), 2, cv2.LINE_AA)

    # Franco
    cy += 80
    cv2.putText(img, cur_franco if cur_franco else "waiting...",
                (15, cy), font, 0.55, (180,180,180), 1, cv2.LINE_AA)

    # Confidence bar
    cy += 30
    cv2.putText(img, f"Confidence: {cur_conf:.1f}%", (10, cy),
                font, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cy += 10
    conf_color = GREEN if cur_conf>=80 else YELLOW if cur_conf>=60 else RED
    draw_progress_bar(img, 10, cy, panel_w-20, 10, cur_conf, 100, conf_color)

    # Buffer progress
    cy += 35
    cv2.putText(img, f"Buffer: {cur_len}/{sequence_len}", (10, cy),
                font, 0.5, (180,180,180), 1, cv2.LINE_AA)
    cy += 10
    buf_color = GREEN if cur_len >= sequence_len else (*GOLD[::-1],)
    draw_progress_bar(img, 10, cy, panel_w-20, 8, cur_len, sequence_len, buf_color)

    # Stats
    cy += 40
    cv2.line(img, (10, cy), (panel_w-10, cy), (50,50,50), 1)
    cy += 20
    cv2.putText(img, f"Total Words: {total}", (10, cy),
                font, 0.5, (*GOLD[::-1],), 1, cv2.LINE_AA)

    # History
    cy += 30
    cv2.putText(img, "HISTORY:", (10, cy), font, 0.48,
                (*GOLD[::-1],), 1, cv2.LINE_AA)
    cy += 5
    for i, word in enumerate(reversed(history[-6:])):
        cy += 22
        alpha_color = (220-i*25, 220-i*25, 220-i*25)
        dot_color   = (*GREEN,) if i==0 else (80,80,80)
        cv2.circle(img, (18, cy-4), 4, dot_color, -1)
        cv2.putText(img, word, (28, cy), font, 0.52,
                    alpha_color, 1, cv2.LINE_AA)

    # ── BOTTOM BAR ──
    draw_filled_rect(img, 0, h-40, w, h, DARK, alpha=0.75)
    hints = "Q: Quit  |  R: Reset  |  +/-: Confidence"
    cv2.putText(img, hints, (panel_w+10, h-12), font, 0.48,
                (120,120,120), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap      = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence = deque(maxlen=SEQUENCE_LEN)
    pred_buf = deque(maxlen=5)
    history  = []
    total    = 0
    conf_thr = CONF_THR

    cur_arabic = "—"
    cur_franco = ""
    cur_conf   = 0.0
    last_ar    = ""

    t_prev = time.time()
    fps    = 0.0

    print("\n✅ الكاميرا شغالة!")
    print("   Q → خروج")
    print("   R → Reset السجل")
    print("   + → رفع الحد الأدنى للثقة")
    print("   - → تخفيض الحد الأدنى للثقة\n")

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ تعذّر فتح الكاميرا")
                break

            # FPS
            now   = time.time()
            fps   = 0.9*fps + 0.1*(1.0/(now-t_prev+1e-6))
            t_prev = now

            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Draw landmarks
            draw_landmarks(img, results)

            # Extract & buffer
            kp = extract_keypoints(results)
            sequence.append(kp)

            # Predict
            if len(sequence) == SEQUENCE_LEN:
                arr  = np.array(sequence, dtype=np.float32)
                norm = (arr - mean) / (std + 1e-8)
                pred = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
                idx  = int(np.argmax(pred))
                conf = float(pred[idx])

                pred_buf.append(idx)
                s_idx  = Counter(pred_buf).most_common(1)[0][0]
                arabic = cls_a[s_idx]
                franco = cls_f[s_idx]

                if conf >= conf_thr:
                    cur_arabic = arabic
                    cur_franco = franco
                    cur_conf   = conf * 100
                    if arabic != last_ar:
                        last_ar = arabic
                        total  += 1
                        if not history or history[-1] != arabic:
                            history.append(arabic)
                            if len(history) > 10:
                                history.pop(0)

                    # Hand box زي الصورة المرجعية
                    box_color = GREEN if conf>=0.8 else YELLOW if conf>=0.6 else RED
                    draw_hand_box(img, results, franco, conf*100, box_color)
                else:
                    cur_conf = conf * 100

            # Draw UI
            draw_ui(img, SEQUENCE_LEN, len(sequence),
                    cur_arabic, cur_franco, cur_conf,
                    history, total, fps)

            # Confidence threshold display
            cv2.putText(img, f"Min Conf: {conf_thr*100:.0f}%",
                        (270, img.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (*GOLD[::-1],), 1)

            cv2.imshow("EGSL - Sign Language Recognition", img)

            # Keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                history.clear()
                total    = 0
                cur_arabic = "—"
                cur_franco = ""
                cur_conf   = 0.0
                last_ar    = ""
                print("🔄 تم الـ Reset")
            elif key == ord('+') or key == ord('='):
                conf_thr = min(0.95, conf_thr + 0.05)
                print(f"⬆ Confidence threshold: {conf_thr*100:.0f}%")
            elif key == ord('-'):
                conf_thr = max(0.30, conf_thr - 0.05)
                print(f"⬇ Confidence threshold: {conf_thr*100:.0f}%")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ البرنامج اتقفل")


if __name__ == "__main__":
    main()
