#   pip install deepface
#   pip install tf-keras -- если выдает отшибку, то дополнительно утснови эту библиотеку

from deepface import DeepFace

def face_verification(img_1, img_2):
    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2)

        return result_dict
    except Exception as _ex:
        return _ex
    
def main():
    face_verification(img_1="faces/yan.jpg", img_2="faces/3model.jpg")

if __name__ == "__main__":
    main()