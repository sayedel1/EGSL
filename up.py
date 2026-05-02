#!/usr/bin/env python3
"""
سكريبت لرفع الملفات إلى Hugging Face Spaces
استخدم: python upload_to_hf.py
"""

from huggingface_hub import HfApi, CommitOperationAdd
import os
import sys

# =====================
# تعديل هذه البيانات
# =====================
HF_TOKEN = "YOUR_HF_TOKEN"  # استبدل بـ Token الخاص بك
REPO_ID = "sayedel1/EGSL"
REPO_TYPE = "space"

# =====================
# قائمة الملفات
# =====================
FILES_TO_UPLOAD = [
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "egsl_model_v2.h5",
    "norm_mean.npy",
    "norm_std.npy",
    "label_map.json",
]

def upload_files():
    """رفع الملفات إلى Hugging Face"""
    
    # التحقق من الـ Token
    if HF_TOKEN == "YOUR_HF_TOKEN":
        print("❌ خطأ: لم تقم بتعديل HF_TOKEN")
        print("اذهب إلى https://huggingface.co/settings/tokens وانسخ الـ Token")
        sys.exit(1)
    
    # إنشاء API client
    api = HfApi()
    
    print(f"🚀 بدء الرفع إلى {REPO_ID}...")
    print("-" * 50)
    
    # التحقق من الملفات
    missing_files = []
    for file in FILES_TO_UPLOAD:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ الملفات التالية غير موجودة:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)
    
    print("✓ جميع الملفات موجودة")
    print("-" * 50)
    
    # رفع الملفات
    uploaded_count = 0
    failed_count = 0
    
    for file in FILES_TO_UPLOAD:
        try:
            file_size = os.path.getsize(file)
            size_mb = file_size / (1024 * 1024)
            
            print(f"📤 جاري رفع {file} ({size_mb:.2f} MB)...", end=" ", flush=True)
            
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=HF_TOKEN,
            )
            
            print("✓")
            uploaded_count += 1
            
        except Exception as e:
            print(f"✗ خطأ: {str(e)}")
            failed_count += 1
    
    print("-" * 50)
    print(f"\n✓ تم رفع {uploaded_count} ملف بنجاح")
    
    if failed_count > 0:
        print(f"✗ فشل رفع {failed_count} ملف")
        sys.exit(1)
    
    print("\n🎉 تم رفع جميع الملفات بنجاح!")
    print(f"الرابط: https://huggingface.co/spaces/{REPO_ID}")
    print("\nالخطوات التالية:")
    print("1. انتظر 5-10 دقائق للبناء")
    print("2. اضغط على 'App' للتحقق من حالة التطبيق")
    print("3. اختبر /health endpoint")

if __name__ == "__main__":
    upload_files()
