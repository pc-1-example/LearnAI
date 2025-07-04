#   pip install deepface
#   pip install tf-keras -- если выдает отшибку, то дополнительно утснови эту библиотеку

from deepface import DeepFace
import json



def face_verification(img_1, img_2):
    try:
        result_dict = DeepFace.verify(img1_path=img_1, img2_path=img_2)

        with open('result.json', 'w') as file:
            json.dump(result_dict, file, indent=4, ensure_ascii=False)

        if result_dict.get("verified"):
            return "Лица совпадают"
        return "Лица не совпадают"

        # return result_dict
    except Exception as _ex:
        return _ex
    


def face_recognition():
    try:
        #result = DeepFace.find(img_path="faces/jaki.jpg", db_path="jeki")
        result = DeepFace.find(img_path="faces/2sam.jpg", db_path="jeki")
        result = result.values.tolist()

        return result
    
    except Exception as _ex:
        return _ex



def face_analysis():
    try:
        result_dict = DeepFace.analyze(img_path="faces/2sam.jpg", actions=['age', 'gender', 'race', 'emotion'])

        with open('face_analysis.json', 'w') as file:
           json.dump(result_dict, file, indent=4, ensure_ascii=False)

        return result_dict
    
    except Exception as _ex:
        return _ex

def main():
    # print(face_verification(img_1="faces/yan.jpg", img_2="faces/3model.jpg"))
    # print(face_verification(img_1="faces/1sam.jpg", img_2="faces/2sam.jpg"))
    # print(face_recognition())
    print(face_analysis())

if __name__ == "__main__":
    main()