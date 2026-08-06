from django.db import models


class Student(models.Model):
    name = models.CharField(max_length=100)
    roll_number = models.CharField(max_length=50, unique=True)
    registered_on = models.DateTimeField(auto_now_add=True)

    # Stores the averaged 128-d face_recognition encoding as a JSON list of
    # floats, computed once capture finishes (see views.save_face). Null
    # until training completes.
    face_encoding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"{self.roll_number} - {self.name}"


class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name="attendance_records")
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)

    class Meta:
        # Prevents marking the same student present twice on the same day
        unique_together = ("student", "date")
        ordering = ["-date", "-time"]

    def __str__(self):
        return f"{self.student.name} - {self.date} {self.time}"