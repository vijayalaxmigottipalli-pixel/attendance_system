from django.db import models
from django.contrib.auth.hashers import make_password, check_password


class Student(models.Model):
    username = models.CharField(
        max_length=100,
        unique=True,
        null=True,
        blank=True
    )

    name = models.CharField(max_length=100)

    roll_number = models.CharField(
        max_length=50,
        unique=True
    )

    registered_on = models.DateTimeField(
        auto_now_add=True
    )

    face_encoding = models.JSONField(
        null=True,
        blank=True
    )

    password = models.CharField(
        max_length=128,
        blank=True,
        null=True
    )

    def set_password(self, raw_password):
        self.password = make_password(raw_password)

    def check_password(self, raw_password):
        if not self.password:
            return False

        return check_password(
            raw_password,
            self.password
        )

    def __str__(self):
        return f"{self.roll_number} - {self.name}"


class ClassSession(models.Model):
    teacher = models.ForeignKey(
        "auth.User",
        on_delete=models.CASCADE
    )

    class_name = models.CharField(
        max_length=100
    )

    subject = models.CharField(
        max_length=100,
        blank=True
    )

    room = models.CharField(
        max_length=100,
        blank=True
    )

    session_code = models.CharField(
        max_length=20,
        unique=True
    )

    started_at = models.DateTimeField(
        auto_now_add=True
    )

    expires_at = models.DateTimeField()

    is_active = models.BooleanField(
        default=True
    )

    def __str__(self):
        return f"{self.class_name} - {self.session_code}"


class AttendanceRecord(models.Model):
    student = models.ForeignKey(
        Student,
        on_delete=models.CASCADE,
        related_name="attendance_records"
    )

    class_session = models.ForeignKey(
        ClassSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="attendance_records"
    )

    date = models.DateField(
        auto_now_add=True
    )

    time = models.TimeField(
        auto_now_add=True
    )

    class Meta:
        unique_together = ("student", "class_session")
        ordering = ["-date", "-time"]

    def __str__(self):
        return f"{self.student.name} - {self.date} {self.time}"