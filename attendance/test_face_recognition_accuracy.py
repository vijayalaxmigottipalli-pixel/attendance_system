"""
Standalone face_recognition (dlib ResNet) accuracy test.
Same dataset and train/test split logic as the earlier LBPH test, so the
two accuracy numbers are directly comparable.

Expected layout:
    kaggle_face_test/
        Dataset.csv              (columns: id,label)
        Faces/Faces/*.jpg        (flat, filenames match the "id" column)

Run:
    python test_face_recognition_accuracy.py
"""

import os
import csv
import random
from collections import defaultdict

import face_recognition
import numpy as np

# --- Config --------------------------------------------------------------
DATASET_DIR = r"C:\Users\vijay\kaggle_face_test"
IMAGES_DIR = os.path.join(DATASET_DIR, "Faces", "Faces")
CSV_PATH = os.path.join(DATASET_DIR, "Dataset.csv")

MIN_IMAGES_PER_PERSON = 15   # same threshold as the LBPH test, for a fair comparison
TEST_FRACTION = 0.2
MATCH_TOLERANCE = 0.6        # face_recognition's default distance threshold for "same person"
RANDOM_SEED = 42
# ---------------------------------------------------------------------------


def load_labels(csv_path):
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = row["label"]
    return labels


def group_by_person(labels):
    groups = defaultdict(list)
    for filename, person in labels.items():
        groups[person].append(filename)
    return groups


def get_face_encoding(image_path):
    """Returns a 128-d face encoding, or None if no face found."""
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if not encodings:
        return None
    return encodings[0]


def main():
    random.seed(RANDOM_SEED)

    print("Loading labels...")
    labels = load_labels(CSV_PATH)
    groups = group_by_person(labels)
    groups = {p: files for p, files in groups.items() if len(files) >= MIN_IMAGES_PER_PERSON}
    print(f"Using {len(groups)} people with >= {MIN_IMAGES_PER_PERSON} images each")

    train_encodings = []   # list of 128-d vectors
    train_labels = []      # parallel list of person names
    test_samples = []      # list of (encoding, true_person)

    print("Encoding faces and building train/test split (this takes a while)...")
    skipped_no_face = 0
    processed = 0

    for person, filenames in groups.items():
        random.shuffle(filenames)
        split_point = max(1, int(len(filenames) * (1 - TEST_FRACTION)))
        train_files = filenames[:split_point]
        test_files = filenames[split_point:]

        for fname in train_files:
            path = os.path.join(IMAGES_DIR, fname)
            enc = get_face_encoding(path)
            processed += 1
            if enc is None:
                skipped_no_face += 1
                continue
            train_encodings.append(enc)
            train_labels.append(person)

        for fname in test_files:
            path = os.path.join(IMAGES_DIR, fname)
            enc = get_face_encoding(path)
            processed += 1
            if enc is None:
                skipped_no_face += 1
                continue
            test_samples.append((enc, person))

        if processed % 200 < len(train_files) + len(test_files):
            print(f"  ...processed ~{processed} images so far")

    print(f"Train encodings: {len(train_encodings)}, Test encodings: {len(test_samples)}, "
          f"skipped (no face detected): {skipped_no_face}")

    if not train_encodings or not test_samples:
        print("Not enough usable images to train/test.")
        return

    train_encodings_np = np.array(train_encodings)

    print("Evaluating on held-out test images...")
    correct = 0
    per_person_correct = defaultdict(int)
    per_person_total = defaultdict(int)
    distances = []

    for encoding, true_person in test_samples:
        face_distances = face_recognition.face_distance(train_encodings_np, encoding)
        best_idx = int(np.argmin(face_distances))
        best_distance = face_distances[best_idx]
        predicted_person = train_labels[best_idx] if best_distance <= MATCH_TOLERANCE else "UNKNOWN"

        distances.append(best_distance)
        per_person_total[true_person] += 1
        if predicted_person == true_person:
            correct += 1
            per_person_correct[true_person] += 1

    accuracy = correct / len(test_samples) * 100
    avg_distance = sum(distances) / len(distances)

    print("\n--- Results ---")
    print(f"Overall accuracy: {accuracy:.1f}% ({correct}/{len(test_samples)})")
    print(f"Average best-match distance (lower = more confident, <0.6 = considered a match): {avg_distance:.3f}")

    print("\nPer-person accuracy (worst 10 shown):")
    per_person_acc = [
        (p, per_person_correct[p] / per_person_total[p] * 100, per_person_total[p])
        for p in per_person_total
    ]
    per_person_acc.sort(key=lambda x: x[1])
    for person, acc, total in per_person_acc[:10]:
        print(f"  {person:30s} {acc:5.1f}%  (n={total})")


if __name__ == "__main__":
    main()