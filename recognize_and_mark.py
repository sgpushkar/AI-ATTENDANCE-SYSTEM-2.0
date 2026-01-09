import cv2
import face_recognition
import pickle
import os

ENCODINGS_PATH = "encodings/known_faces.pkl"


def recognize_from_image(image_path):
    """
    Takes image path
    Returns list of recognized student folder names
    """

    if not os.path.exists(ENCODINGS_PATH):
        raise Exception("Face encodings not found")

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_names = data["names"]

    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    recognized_students = set()

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)

        if True in matches:
            matched_idxs = [i for i, b in enumerate(matches) if b]
            counts = {}

            for i in matched_idxs:
                name = known_names[i]
                counts[name] = counts.get(name, 0) + 1

            recognized_name = max(counts, key=counts.get)
            recognized_students.add(recognized_name)

    return list(recognized_students)
