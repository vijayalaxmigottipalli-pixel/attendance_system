import os
import base64
import json
from datetime import datetime

import cv2
import numpy as np
import face_recognition

from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone

from .models import Student, AttendanceRecord


FACES_DIR = os.path.join(settings.BASE_DIR, "face_dataset")
IMAGES_PER_STUDENT = 30
MATCH_TOLERANCE = 0.6  # face_recognition's standard distance threshold for "same person"

import re

ROLL_NUMBER_PATTERN = re.compile(r"^\d{2}B91[A-Z0-9]{4,6}$")
ROLL_NUMBER_EXAMPLE = "24B91A5471"


def home(request):
    students = Student.objects.all().order_by("-registered_on")
    return render(request, "attendance/home.html", {"students": students})


def register_student(request):

    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        roll_number = request.POST.get("roll_number", "").strip().upper()

        if not name or not roll_number:
            return render(request, "attendance/register.html", {
                "error": "Both name and roll number are required.",
                "roll_example": ROLL_NUMBER_EXAMPLE,
            })

        if not ROLL_NUMBER_PATTERN.match(roll_number):
            return render(request, "attendance/register.html", {
                "error": f"Roll number must be in the format {ROLL_NUMBER_EXAMPLE} (e.g. year + B91 + your ID).",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "name": name,
            })

        if Student.objects.filter(roll_number=roll_number).exists():
            return render(request, "attendance/register.html", {
                "error": f"Roll number {roll_number} is already registered.",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "name": name,
            })

        student = Student.objects.create(name=name, roll_number=roll_number)
        return redirect("capture_page", student_id=student.id)

    return render(request, "attendance/register.html", {
        "roll_example": ROLL_NUMBER_EXAMPLE,
    })


def capture_page(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    return render(request, "attendance/capture.html", {
        "student": student,
        "images_needed": IMAGES_PER_STUDENT
    })


@csrf_exempt
def save_face(request, student_id):
    """
    Receives a single webcam frame (base64 JPEG) from the browser, detects
    a face using face_recognition, crops and saves it to disk. Once
    IMAGES_PER_STUDENT images have been captured, computes an averaged
    face encoding across all of them and stores it on the Student.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"}, status=405)

    student = get_object_or_404(Student, id=student_id)

    try:
        data = json.loads(request.body)
        image_data = data.get("image", "")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return JsonResponse({"status": "no_face", "message": "Could not decode image"})

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)

        if not face_locations:
            return JsonResponse({"status": "no_face", "message": "No face detected"})

        def area(loc):
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        top, right, bottom, left = max(face_locations, key=area)

        h, w, _ = frame_bgr.shape
        pad = 20
        top = max(0, top - pad)
        left = max(0, left - pad)
        bottom = min(h, bottom + pad)
        right = min(w, right + pad)

        face_crop_bgr = frame_bgr[top:bottom, left:right]
        face_crop_bgr = cv2.resize(face_crop_bgr, (200, 200))

        student_dir = os.path.join(FACES_DIR, str(student.id))
        os.makedirs(student_dir, exist_ok=True)

        existing_count = len(os.listdir(student_dir))
        new_count = existing_count + 1
        save_path = os.path.join(student_dir, f"{new_count}.jpg")
        cv2.imwrite(save_path, face_crop_bgr)

        response = {"status": "ok", "count": new_count}

        if new_count >= IMAGES_PER_STUDENT:
            encoding = _build_average_encoding(student_dir)
            if encoding is not None:
                student.face_encoding = encoding.tolist()
                student.save()
                response["training_complete"] = True
            else:
                response["training_complete"] = False
                response["message"] = (
                    "Captured all images but couldn't compute a face encoding "
                    "from any of them — try recapturing with better lighting."
                )

        return JsonResponse(response)

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def _build_average_encoding(student_dir):
    """
    Reads every saved face crop for a student, computes a face_recognition
    encoding for each, and returns the average as a numpy array. Returns
    None if no usable encodings were found.
    """
    encodings = []

    for filename in os.listdir(student_dir):
        path = os.path.join(student_dir, filename)
        image = face_recognition.load_image_file(path)
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])

    if not encodings:
        return None

    return np.mean(encodings, axis=0)


def recognize_page(request):
    """
    Kiosk-mode attendance page: webcam runs continuously, recognized
    students are marked present and appended to an on-screen list.
    """
    return render(request, "attendance/recognize.html")


@csrf_exempt
def mark_attendance(request):
    """
    Receives a single webcam frame, finds the closest matching trained
    student (if any), and marks them present for today. Designed to be
    called repeatedly (e.g. every ~1.5s) from a continuously-running
    webcam page.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        image_data = data.get("image", "")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            return JsonResponse({"status": "no_face", "message": "Could not decode image"})

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(frame_rgb)

        if not face_locations:
            return JsonResponse({"status": "no_face"})

        # Only look at the largest face in frame (assume one person at a time)
        def area(loc):
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        best_location = max(face_locations, key=area)
        frame_encodings = face_recognition.face_encodings(frame_rgb, known_face_locations=[best_location])

        if not frame_encodings:
            return JsonResponse({"status": "no_face"})

        frame_encoding = frame_encodings[0]

        trained_students = Student.objects.exclude(face_encoding__isnull=True)

        if not trained_students.exists():
            return JsonResponse({"status": "no_trained_students",
                                  "message": "No students have completed face training yet."})

        known_encodings = np.array([s.face_encoding for s in trained_students])
        student_list = list(trained_students)

        distances = face_recognition.face_distance(known_encodings, frame_encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        if best_distance > MATCH_TOLERANCE:
            return JsonResponse({"status": "unrecognized", "distance": float(best_distance)})

        matched_student = student_list[best_idx]
        today = timezone.localdate()

        record, created = AttendanceRecord.objects.get_or_create(
            student=matched_student,
            date=today,
        )

        return JsonResponse({
            "status": "marked" if created else "already_marked",
            "student_id": matched_student.id,
            "student_name": matched_student.name,
            "roll_number": matched_student.roll_number,
            "distance": float(best_distance),
            "time": record.time.strftime("%I:%M %p"),
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


def attendance_report(request):
    """
    Shows every registered student with Present/Absent status for a given
    date (defaults to today), plus summary counts. Pass ?date=YYYY-MM-DD
    in the query string to view a different day.
    """
    date_str = request.GET.get("date")
    if date_str:
        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            selected_date = timezone.localdate()
    else:
        selected_date = timezone.localdate()

    students = Student.objects.all().order_by("roll_number")
    records = AttendanceRecord.objects.filter(date=selected_date).select_related("student")
    records_by_student_id = {r.student_id: r for r in records}

    rows = []
    present_count = 0
    for student in students:
        record = records_by_student_id.get(student.id)
        present = record is not None
        if present:
            present_count += 1
        rows.append({
            "student": student,
            "present": present,
            "time": record.time.strftime("%I:%M %p") if record else None,
        })

    total_count = students.count()
    absent_count = total_count - present_count

    return render(request, "attendance/report.html", {
        "rows": rows,
        "selected_date": selected_date,
        "total_count": total_count,
        "present_count": present_count,
        "absent_count": absent_count,
    })