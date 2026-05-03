"""
EGSL — Egyptian Sign Language Recognition
إصدار يعمل مع العربية وبالاتجاه الصحيح (RTL)
"""

import cv2
import mediapipe as mp
import numpy as np
import json, os, time
import tensorflow as tf
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

# إضافات لمعالجة اتجاه النص العربي
import arabic_reshaper
from bidi.algorithm import get_display

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
#  COLORS
# ─────────────────────────────────────────────
GOLD   = (76, 168, 201)
GREEN  = (113, 237, 46)
RED    = (87, 71, 255)
YELLOW = (0, 215, 255)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
DARK   = (16, 16, 24)

# ─────────────────────────────────────────────
#  FUNCTIONS FOR ARABIC TEXT (CORRECT ORDER)
# ─────────────────────────────────────────────
def reshape_arabic(text):
    """
    يعيد تشكيل النص العربي ليظهر بالاتجاه الصحيح 
    (الحروف متصلة ومن اليمين لليسار)
    """
    if not text or text == "—":
        return text
    reshaped = arabic_reshaper.reshape(text)   # يصل الحروف
    bidi_text = get_display(reshaped)          # يحدد اتجاه RTL
    return bidi_text

def get_arabic_font(size=40):
    """إرجاع خط عربي مناسب للنظام (يفضل الخطوط التي تدعم العربية)"""
    possible_fonts = [
        "C:/Windows/Fonts/arial.ttf",          # ويندوز
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arialuni.ttf",
        "/System/Library/Fonts/Helvetica.ttc", # ماك
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # لينكس
        "arial.ttf"
    ]
    for font_path in possible_fonts:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    return ImageFont.load_default()  # الحل الأخير (قد لا يدعم العربية)

def draw_arabic_text(img, text, x, y, color=WHITE, font=None, size=42):
    """
    رسم نص عربي على صورة OpenCV بعد إعادة تشكيله ليكون متصلاً وفي الاتجاه الصحيح.
    """
    if not text or text == "—":
        return img
    if font is None:
        font = get_arabic_font(size)
    
    # إعادة تشكيل النص
    fixed_text = reshape_arabic(text)
    
    # تحويل الصورة من BGR إلى RGB للرسم باستخدام PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    # PIL تستخدم ألوان (R,G,B) بينما OpenCV (B,G,R)
    draw.text((x, y), fixed_text, font=font, fill=(color[2], color[1], color[0]))
    # العودة إلى BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
#  LOAD MODEL & LABELS
# ─────────────────────────────────────────────
print("⏳ جاري تحميل الموديل...")
model = tf.keras.models.load_model(MODEL_PATH)
mean  = np.load(MEAN_PATH)
std   = np.load(STD_PATH)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    lm = json.load(f)

# بناء قائمة الكلمات
if "franco" in lm:
    n = len(lm["franco"])
    cls_f = [""]*n
    cls_a = [""]*n
    for name, idx in lm["franco"].items():
        cls_f[idx] = name
        cls_a[idx] = lm.get("arabic", {}).get(name, FRANCO_TO_ARABIC.get(name, name))
else:
    n = len(lm)
    cls_f = [""]*n
    cls_a = [""]*n
    for name, idx in lm.items():
        cls_f[idx] = name
        cls_a[idx] = FRANCO_TO_ARABIC.get(name, name)

print(f"✅ الموديل جاهز — {n} كلمة")
print(f"📝 مثال: {cls_f[0]} → {cls_a[0]}")

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

def extract_keypoints(results):
    pose = np.array([[lm.x,lm.y,lm.z,lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*4)
    lh   = np.array([[lm.x,lm.y,lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[lm.x,lm.y,lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=GOLD, thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(150,210,240), thickness=1))
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            mp_drawing.draw_landmarks(image, hand, mp_holistic.HAND_CONNECTIONS,
                                      mp_draw_styles.get_default_hand_landmarks_style(),
                                      mp_draw_styles.get_default_hand_connections_style())
    return image

# ─────────────────────────────────────────────
#  UI DRAWING
# ─────────────────────────────────────────────
def draw_ui(img, seq_len, cur_len, cur_arabic, cur_franco,
            cur_conf, history, total, fps):
    h, w = img.shape[:2]
    font_small = cv2.FONT_HERSHEY_SIMPLEX

    # الشريط العلوي
    cv2.rectangle(img, (0,0), (w,55), DARK, -1)
    cv2.putText(img, "EGSL - Egyptian Sign Language", (12,35), font_small, 0.75, GOLD, 2)
    fw, _ = cv2.getTextSize(f"FPS: {fps:.1f}", font_small, 0.6, 1)[0]
    cv2.putText(img, f"FPS: {fps:.1f}", (w-fw-12,35), font_small, 0.6, GREEN, 1)

    # اللوحة الجانبية اليسرى
    panel_w = 260
    overlay = img.copy()
    cv2.rectangle(overlay, (0,55), (panel_w,h), DARK, -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    cv2.line(img, (panel_w,55), (panel_w,h), GOLD, 1)

    # عنوان "PREDICTION"
    cv2.putText(img, "PREDICTION", (10,100), font_small, 0.5, GOLD, 1)
    cv2.line(img, (10,105), (panel_w-10,105), GOLD, 1)

    # الكلمة العربية (باستخدام الدالة المصححة)
    if cur_arabic and cur_arabic != "—":
        img = draw_arabic_text(img, cur_arabic, 15, 150, WHITE, size=42)
    else:
        cv2.putText(img, "---", (15,150), font_small, 1.0, (150,150,150), 2)

    # الكلمة بالفرانكو
    cv2.putText(img, cur_franco if cur_franco else "waiting...", (15,200),
                font_small, 0.55, (180,180,180), 1)

    # نسبة الثقة
    cv2.putText(img, f"Confidence: {cur_conf:.1f}%", (10,240), font_small, 0.5, (180,180,180), 1)
    bar_x, bar_y, bar_w, bar_h = 10, 250, panel_w-20, 10
    cv2.rectangle(img, (bar_x,bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), -1)
    filled = int(bar_w * min(cur_conf/100, 1.0))
    conf_color = GREEN if cur_conf>=80 else YELLOW if cur_conf>=60 else RED
    if filled>0:
        cv2.rectangle(img, (bar_x,bar_y), (bar_x+filled, bar_y+bar_h), conf_color, -1)

    # تقدم المخزن المؤقت
    cv2.putText(img, f"Buffer: {cur_len}/{seq_len}", (10,280), font_small, 0.5, (180,180,180), 1)
    cv2.rectangle(img, (10,290), (panel_w-10,300), (50,50,50), -1)
    filled_buf = int((panel_w-20) * (cur_len/seq_len))
    if filled_buf>0:
        cv2.rectangle(img, (10,290), (10+filled_buf,300), GOLD, -1)

    # سجل الكلمات (HISTORY) مع تشكيل عربي
    cv2.putText(img, "HISTORY:", (10,330), font_small, 0.48, GOLD, 1)
    y_hist = 350
    for i, word in enumerate(reversed(history[-6:])):
        cv2.circle(img, (18, y_hist-4), 4, GREEN if i==0 else (80,80,80), -1)
        # استخدام draw_arabic_text لكل كلمة في التاريخ
        img = draw_arabic_text(img, word, 28, y_hist-4, (220-i*25,220-i*25,220-i*25), size=22)
        y_hist += 22

    cv2.putText(img, f"Total Words: {total}", (10, y_hist+10), font_small, 0.5, GOLD, 1)

    # الشريط السفلي
    cv2.rectangle(img, (0, h-40), (w, h), DARK, -1)
    hints = "Q: Quit  |  R: Reset  |  +/-: Confidence"
    cv2.putText(img, hints, (panel_w+10, h-12), font_small, 0.48, (120,120,120), 1)
    cv2.putText(img, f"Min Conf: {CONF_THR*100:.0f}%", (panel_w+10, 80), font_small, 0.48, GOLD, 1)

    return img

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sequence = deque(maxlen=SEQUENCE_LEN)
    pred_buf = deque(maxlen=5)
    history = []
    total = 0
    conf_thr = CONF_THR

    cur_arabic = "—"
    cur_franco = ""
    cur_conf = 0.0
    last_ar = ""

    t_prev = time.time()
    fps = 0.0

    print("\n✅ الكاميرا شغالة! اضغط Q للخروج، R لإعادة التعيين، +/- لتغيير حد الثقة")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ تعذّر فتح الكاميرا")
                break

            now = time.time()
            fps = 0.9*fps + 0.1*(1.0/(now-t_prev+1e-6))
            t_prev = now

            img = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            draw_landmarks(img, results)

            kp = extract_keypoints(results)
            sequence.append(kp)

            if len(sequence) == SEQUENCE_LEN:
                arr = np.array(sequence, dtype=np.float32)
                norm = (arr - mean) / (std + 1e-8)
                pred = model.predict(np.expand_dims(norm, 0), verbose=0)[0]
                idx = np.argmax(pred)
                conf = float(pred[idx])

                pred_buf.append(idx)
                most_common_idx = Counter(pred_buf).most_common(1)[0][0]
                arabic = cls_a[most_common_idx]
                franco = cls_f[most_common_idx]

                if conf >= conf_thr:
                    cur_arabic = arabic
                    cur_franco = franco
                    cur_conf = conf * 100
                    if arabic != last_ar:
                        last_ar = arabic
                        total += 1
                        if not history or history[-1] != arabic:
                            history.append(arabic)
                            if len(history) > 10:
                                history.pop(0)
                else:
                    cur_conf = conf * 100

            img = draw_ui(img, SEQUENCE_LEN, len(sequence), cur_arabic, cur_franco,
                          cur_conf, history, total, fps)

            cv2.imshow("EGSL - Sign Language Recognition", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                history.clear()
                total = 0
                cur_arabic = "—"
                cur_franco = ""
                cur_conf = 0.0
                last_ar = ""
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
