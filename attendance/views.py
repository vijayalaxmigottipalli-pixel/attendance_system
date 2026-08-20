import os
import base64
import json
from datetime import datetime, timedelta

import cv2
import numpy as np
import face_recognition


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils import timezone
from functools import wraps
from django.contrib.auth.hashers import check_password



from .models import Student, AttendanceRecord, ClassSession


FACES_DIR = os.path.join(settings.BASE_DIR, "face_dataset")
IMAGES_PER_STUDENT = 30
MATCH_TOLERANCE = 0.6  # face_recognition's standard distance threshold for "same person"

import re

ROLL_NUMBER_PATTERN = re.compile(r"^\d{2}B91[A-Z0-9]{4,6}$")
ROLL_NUMBER_EXAMPLE = "24B91A5471"


def student_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        student_id = request.session.get("student_id")
        if not student_id:
            return redirect("student_login")
        student = Student.objects.filter(id=student_id).first()
        if not student:
            del request.session["student_id"]
            return redirect("student_login")
        request.student = student
        return view_func(request, *args, **kwargs)
    return wrapper


def student_or_teacher_required(view_func):
    """
    Guards the profile page: lets in a logged-in teacher (any teacher can
    view any student's profile) OR the student themself viewing their own
    profile. Anyone else is redirected to login.
    """
    @wraps(view_func)
    def wrapper(request, student_id, *args, **kwargs):
        is_teacher = request.user.is_authenticated
        is_this_student = request.session.get("student_id") == student_id
        if not (is_teacher or is_this_student):
            return redirect("login")
        return view_func(request, student_id, *args, **kwargs)
    return wrapper


def _attendance_stats(student):
    """
    Shared helper: total periods ever held (across all class sessions),
    how many the student attended, and the resulting percentage.
    Used by both the dashboard and the profile page so the numbers
    always match.
    """
    attendance_records = AttendanceRecord.objects.filter(
        student=student
    ).select_related("class_session").order_by("-date", "-time")

    total_classes = ClassSession.objects.count()
    total_present = attendance_records.count()
    attendance_percentage = (
        round((total_present / total_classes) * 100, 1) if total_classes > 0 else None
    )

    return attendance_records, total_classes, total_present, attendance_percentage


def home(request):
    students = Student.objects.all().order_by("-registered_on")
    return render(request, "attendance/home.html", {"students": students})


# ---------------------------------------------------------------------
# Registration (public — Student or Teacher, picked via role toggle)
# ---------------------------------------------------------------------

def register(request):
    """
    Public registration page. Anyone can sign up as either a Student
    or a Teacher — the 'role' hidden field (set by the tab toggle in
    the template) decides which branch runs.
    """
    if request.method == "POST":
        role = request.POST.get("role", "student")

        if role == "teacher":
            name = request.POST.get("name", "").strip()
            username = request.POST.get("username", "").strip()
            password = request.POST.get("password", "").strip()

            if not name or not username:
                return render(request, "attendance/register.html", {
                    "error": "Name and username are required.",
                    "roll_example": ROLL_NUMBER_EXAMPLE,
                    "role": role,
                    "name": name,
                    "username": username,
                })

            if not password or len(password) < 4:
                return render(request, "attendance/register.html", {
                    "error": "Please set a password (at least 4 characters).",
                    "roll_example": ROLL_NUMBER_EXAMPLE,
                    "role": role,
                    "name": name,
                    "username": username,
                })

            if User.objects.filter(username=username).exists():
                return render(request, "attendance/register.html", {
                    "error": f"Username '{username}' is already taken.",
                    "roll_example": ROLL_NUMBER_EXAMPLE,
                    "role": role,
                    "name": name,
                })

            user = User.objects.create_user(
                username=username, password=password, first_name=name
            )
            user.is_staff = True  # needed for start_class's is_staff check
            user.save()
            login(request, user)
            return redirect("teacher_dashboard")

        # role == "student"
        name = request.POST.get("name", "").strip()
        student_username = request.POST.get("student_username", "").strip()
        roll_number = request.POST.get("roll_number", "").strip().upper()
        password = request.POST.get("password", "").strip()

        if not name or not roll_number or not student_username:
            return render(request, "attendance/register.html", {
                "error": "Name, username, and roll number are all required.",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "role": role,
                "name": name,
                "username": student_username,
            })

        if not password or len(password) < 4:
            return render(request, "attendance/register.html", {
                "error": "Please set a password (at least 4 characters).",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "role": role,
                "name": name,
                "username": student_username,
                "roll_number": roll_number,
            })

        if not ROLL_NUMBER_PATTERN.match(roll_number):
            return render(request, "attendance/register.html", {
                "error": f"Roll number must be in the format {ROLL_NUMBER_EXAMPLE} (e.g. year + B91 + your ID).",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "role": role,
                "name": name,
                "username": student_username,
            })

        if Student.objects.filter(roll_number=roll_number).exists():
            return render(request, "attendance/register.html", {
                "error": f"Roll number {roll_number} is already registered.",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "role": role,
                "name": name,
                "username": student_username,
            })

        if Student.objects.filter(username=student_username).exists():
            return render(request, "attendance/register.html", {
                "error": f"Username '{student_username}' is already taken.",
                "roll_example": ROLL_NUMBER_EXAMPLE,
                "role": role,
                "name": name,
                "roll_number": roll_number,
            })

        student = Student(name=name, username=student_username, roll_number=roll_number)
        student.set_password(password)
        student.save()

        # Log the new student into their session immediately, so the
        # capture page (next step) recognizes them without needing a
        # separate login step.
        request.session["student_id"] = student.id

        return redirect("capture_page", student_id=student.id)

    return render(request, "attendance/register.html", {
        "roll_example": ROLL_NUMBER_EXAMPLE,
    })


def capture_page(request, student_id):
    """
    Face-capture page. Accessible either by the student themself
    (right after registering, via their session) or by a teacher
    (in case a teacher is capturing faces on a student's behalf).
    """
    student = get_object_or_404(Student, id=student_id)

    is_this_student = request.session.get("student_id") == student.id
    is_teacher = request.user.is_authenticated

    if not (is_this_student or is_teacher):
        return redirect("login")

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

        # Detect on a downscaled copy for speed, then scale coords back up
        small_rgb = cv2.resize(frame_rgb, (0, 0), fx=0.5, fy=0.5)
        face_locations_small = face_recognition.face_locations(small_rgb)

        if not face_locations_small:
            return JsonResponse({"status": "no_face", "message": "No face detected"})

        def area(loc):
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        top, right, bottom, left = max(face_locations_small, key=area)
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2

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
                refresh_student_cache()
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


@login_required
def recognize_page(request):
    """
    Kiosk-mode attendance page: webcam runs continuously, recognized
    students are marked present and appended to an on-screen list.
    """
    return render(request, "attendance/recognize.html")


@student_required
def student_recognize_page(request):
    """
    Student-facing version of the recognize page. Requires a valid,
    unexpired session code to already be stored in their session —
    otherwise they're sent to enter one first.
    """
    if not request.session.get("active_session_code"):
        return redirect("enter_session_code")

    return render(request, "attendance/recognize.html", {
        "student_mode": True,
        "student": request.student,
    })


def _match_student(frame_encoding, known_encodings, student_list):
    """
    Compares a single face encoding against every trained student's
    encoding and returns (matched_student_or_None, best_distance).
    """
    distances = face_recognition.face_distance(known_encodings, frame_encoding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])

    if best_distance > MATCH_TOLERANCE:
        return None, best_distance

    return student_list[best_idx], best_distance


# In-memory cache of trained student encodings, so recognize/ doesn't hit
# the DB and re-parse JSON on every single poll (every ~1.5s). Invalidated
# whenever save_face finishes training a new student (see _build_average_encoding
# call site) or manually via refresh_student_cache().
_student_cache = {"encodings": None, "students": None}


def _get_known_students():
    if _student_cache["encodings"] is None:
        trained_students = list(Student.objects.exclude(face_encoding__isnull=True))
        if trained_students:
            _student_cache["encodings"] = np.array([s.face_encoding for s in trained_students])
            _student_cache["students"] = trained_students
        else:
            _student_cache["encodings"] = np.empty((0, 128))
            _student_cache["students"] = []
    return _student_cache["encodings"], _student_cache["students"]


def refresh_student_cache():
    """Call this after a new student finishes training so recognize/ picks
    them up immediately instead of waiting for a server restart."""
    _student_cache["encodings"] = None
    _student_cache["students"] = None


# Detection is run on the frame scaled down to this width (px). Lower =
# faster detection but can miss small/far-away faces. 480-640 is a good
# balance for a webcam kiosk where people stand close to the camera.
DETECTION_MAX_WIDTH = 480


@csrf_exempt
def mark_attendance(request):
    """
    Receives a single webcam frame and marks attendance.

    Student flow (mode="single"): trusts only the session_code stored
    server-side in request.session (set by enter_session_code after
    validating it) plus the logged-in student's own face — never the
    code alone, and never a face that doesn't belong to that student.

    Teacher kiosk flow (mode="multiple"): unchanged — any active class
    session, any recognized face.
    """
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        mode = data.get("mode", "multiple")
        image_data = data.get("image", "")

        logged_in_student = None

        if mode == "single":
            # 1) Logged-in student required
            student_id = request.session.get("student_id")
            if not student_id:
                return JsonResponse({
                    "status": "error",
                    "message": "You must be logged in to mark attendance."
                }, status=403)

            logged_in_student = Student.objects.filter(id=student_id).first()
            if not logged_in_student:
                return JsonResponse({"status": "error", "message": "Student not found."}, status=403)

            # 2) Valid session code required — read from the student's own
            # server-side session, set only by enter_session_code after a
            # real validation. The client can't spoof this.
            active_code = request.session.get("active_session_code")
            active_session = ClassSession.objects.filter(
                session_code=active_code,
                is_active=True,
                expires_at__gt=timezone.now()
            ).first() if active_code else None

            if not active_session:
                return JsonResponse({
                    "status": "invalid_session",
                    "message": "Your session code has expired. Please enter it again."
                }, status=403)

        else:
            # Teacher kiosk mode: any currently active class session works.
            active_session = ClassSession.objects.filter(
                is_active=True,
                expires_at__gt=timezone.now()
            ).first()

            if not active_session:
                return JsonResponse({
                    "status": "no_active_session",
                    "message": "No active class session. Attendance cannot be marked."
                }, status=403)

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

        def area(loc):
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        if mode == "single":
            face_locations = [max(face_locations, key=area)]

        frame_encodings = face_recognition.face_encodings(
            frame_rgb, known_face_locations=face_locations
        )

        if not frame_encodings:
            return JsonResponse({"status": "no_face"})

        if mode == "single":
            # 3) Face verification — compare only against the logged-in
            # student's own stored encoding. The code + login alone are
            # never enough; the face has to match too.
            if logged_in_student.face_encoding is None:
                return JsonResponse({
                    "status": "error",
                    "message": "Your face hasn't been registered yet."
                })
            known_encodings = np.array([logged_in_student.face_encoding])
            student_list = [logged_in_student]
        else:
            trained_students = Student.objects.exclude(face_encoding__isnull=True)
            if not trained_students.exists():
                return JsonResponse({"status": "no_trained_students",
                                      "message": "No students have completed face training yet."})
            known_encodings = np.array([s.face_encoding for s in trained_students])
            student_list = list(trained_students)

        today = timezone.localdate()
        results = []
        seen_student_ids = set()

        for frame_encoding in frame_encodings:
            matched_student, best_distance = _match_student(
                frame_encoding, known_encodings, student_list
            )

            if matched_student is None:
                results.append({"status": "unrecognized", "distance": best_distance})
                continue

            if mode == "single" and matched_student.id != logged_in_student.id:
                # Face present but didn't match the logged-in student closely
                # enough — don't let anyone else's face mark this student present.
                results.append({"status": "unrecognized", "distance": best_distance})
                continue

            if matched_student.id in seen_student_ids:
                continue
            seen_student_ids.add(matched_student.id)

            record, created = AttendanceRecord.objects.get_or_create(
                student=matched_student,
                class_session=active_session,
            )

            results.append({
                "status": "marked" if created else "already_marked",
                "student_id": matched_student.id,
                "student_name": matched_student.name,
                "roll_number": matched_student.roll_number,
                "distance": best_distance,
                "time": record.time.strftime("%I:%M %p"),
            })

        return JsonResponse({
            "status": "ok",
            "faces_detected": len(face_locations),
            "results": results,
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@login_required
def attendance_report(request):
    """
    Shows attendance broken down by class period (ClassSession) for a
    given date (defaults to today) — since a student can now attend
    multiple periods in the same day, each period gets its own roster
    with Present/Absent status. Pass ?date=YYYY-MM-DD to view another day.
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
    total_count = students.count()

    sessions = ClassSession.objects.filter(
        started_at__date=selected_date
    ).order_by("started_at")

    periods = []
    for session in sessions:
        session_records = AttendanceRecord.objects.filter(
            class_session=session
        ).select_related("student")
        records_by_student_id = {r.student_id: r for r in session_records}

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

        periods.append({
            "session": session,
            "rows": rows,
            "present_count": present_count,
            "absent_count": total_count - present_count,
        })

    return render(request, "attendance/report.html", {
        "periods": periods,
        "selected_date": selected_date,
        "total_count": total_count,
    })


# ---------------------------------------------------------------------
# Login (public — Student or Teacher, picked via role toggle)
# ---------------------------------------------------------------------

def login_view(request):
    if request.session.get("student_id"):
        return redirect("student_dashboard")
    if request.user.is_authenticated:
        return redirect("teacher_dashboard")

    if request.method == "POST":
        role = request.POST.get("role", "student")

        if role == "teacher":
            username = request.POST.get("username", "").strip()
            password = request.POST.get("password", "")
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("teacher_dashboard")
            return render(request, "attendance/login.html", {
                "error": "Invalid username or password.",
                "role": role,
                "username": username,
            })

        # role == "student" — logs in by username now (roll_number kept as fallback)
        student_username = request.POST.get("student_username", "").strip()
        password = request.POST.get("password", "")

        student = Student.objects.filter(username=student_username).first()
        if not student:
            # fallback: allow the old roll-number-based login to keep working
            student = Student.objects.filter(roll_number=student_username.upper()).first()

        if student and student.check_password(password):
            request.session["student_id"] = student.id
            return redirect("student_dashboard")

        return render(request, "attendance/login.html", {
            "error": "Invalid username or password.",
            "role": role,
            "student_username": student_username,
        })

    role = request.GET.get("role", "student")
    return render(request, "attendance/login.html", {"role": role})


def teacher_login(request):
    """Old URL kept alive as a redirect so existing links don't break."""
    return redirect("/?role=teacher")


def student_login(request):
    """Old URL kept alive as a redirect so existing links don't break."""
    return redirect("/?role=student")


def teacher_logout(request):
    logout(request)
    return redirect("login")


def teacher_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect('login')
        return view_func(request, *args, **kwargs)
    return wrapper


@login_required
def teacher_dashboard(request):
    """
    Landing page for a logged-in teacher: quick stats plus links to
    register students, start a class session, and view reports.
    """
    total_students = Student.objects.count()
    trained_students = Student.objects.exclude(face_encoding__isnull=True).count()

    today = timezone.localdate()
    present_today = AttendanceRecord.objects.filter(date=today).count()

    active_session = ClassSession.objects.filter(
        is_active=True,
        expires_at__gt=timezone.now()
    ).first()

    return render(request, "attendance/teacher_dashboard.html", {
        "teacher_name": request.user.get_full_name() or request.user.username,
        "total_students": total_students,
        "trained_students": trained_students,
        "present_today": present_today,
        "total_today": Student.objects.count(),
        "active_session": active_session,
    })


@login_required
def start_class(request):
    if not request.user.is_staff:
        return redirect("login")

    if request.method == "POST":
        import secrets

        class_name = request.POST.get("class_name", "").strip()
        subject = request.POST.get("subject", "").strip()
        room = request.POST.get("room", "").strip()

        # Make sure all fields are provided
        if not class_name or not subject or not room:
            return render(request, "attendance/start_class.html", {
                "error": "Please select the class, subject, and room."
            })

        # Generate a new unique session code
        session_code = secrets.token_hex(4).upper()

        # Close any previous active session for this teacher
        ClassSession.objects.filter(
            teacher=request.user,
            is_active=True
        ).update(is_active=False)

        # Create the new class session
        session = ClassSession.objects.create(
            teacher=request.user,
            class_name=class_name,
            subject=subject,
            room=room,
            session_code=session_code,
            expires_at=timezone.now() + timedelta(minutes=2),
            is_active=True,
        )

        return redirect("view_class_session", session_code=session_code)

    return render(request, "attendance/start_class.html")


@login_required
def view_class_session(request, session_code):
    """
    Displays an already-created class session (by code). Reached via
    redirect from start_class, so refreshing this page is a plain GET —
    it re-shows the same session instead of creating a new one.
    """
    session = get_object_or_404(
        ClassSession, session_code=session_code, teacher=request.user
    )
    return render(request, "attendance/class_session.html", {
        "session": session,
    })


# ---------------------------------------------------------------------
# Student authentication
# ---------------------------------------------------------------------

def student_logout(request):
    request.session.pop("student_id", None)
    return redirect("login")


@student_required
def student_dashboard(request):
    student_id = request.session.get("student_id")

    if not student_id:
        return redirect("student_login")

    student = get_object_or_404(Student, id=student_id)

    attendance_records, total_classes, total_present, attendance_percentage = _attendance_stats(student)

    return render(request, "attendance/student_dashboard.html", {
        "student": student,
        "attendance_records": attendance_records,
        "total_classes": total_classes,
        "total_present": total_present,
        "attendance_percentage": attendance_percentage,
    })


@student_or_teacher_required
def student_profile(request, student_id):
    """
    Read-only profile page for a single student — shows identity details
    plus attendance percentage and full history. Viewable by the student
    themself, or by any logged-in teacher (e.g. clicking a name in the
    Report page).
    """
    student = get_object_or_404(Student, id=student_id)

    attendance_records, total_classes, total_present, attendance_percentage = _attendance_stats(student)

    return render(request, "attendance/profile.html", {
        "student": student,
        "attendance_records": attendance_records,
        "total_classes": total_classes,
        "total_present": total_present,
        "attendance_percentage": attendance_percentage,
        "is_teacher_view": request.user.is_authenticated,
    })


# ---------------------------------------------------------------------
# Session-code attendance flow (student)
# ---------------------------------------------------------------------

@student_required
def enter_session_code(request):
    """
    Student enters the teacher's session code here first. On success, the
    validated code is stored in the student's own Django session (server-side,
    not something the browser can forge) so mark_attendance can trust it later.
    """
    if request.method == "POST":
        code = request.POST.get("session_code", "").strip().upper()

        session = ClassSession.objects.filter(
            session_code=code,
            is_active=True,
            expires_at__gt=timezone.now()
        ).first()

        if not session:
            return render(request, "attendance/enter_code.html", {
                "error": "Invalid or expired session code.",
                "session_code": code,
            })

        request.session["active_session_code"] = session.session_code
        return redirect("student_recognize_page")

    return render(request, "attendance/enter_code.html")


@login_required
def end_session(request):
    if request.method == "POST":
        ClassSession.objects.filter(
            teacher=request.user,
            is_active=True
        ).update(is_active=False)
    return redirect("teacher_dashboard")