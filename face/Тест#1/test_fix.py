from deepface import DeepFace
import os, traceback, json

def face_verification(img1, img2):
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            detector_backend="retinaface",
            enforce_detection=True,       
            silent=False                  
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Verification failed:", e)
        traceback.print_exc()
