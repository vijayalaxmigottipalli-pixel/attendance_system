from django.contrib import admin
from .models import Student, AttendanceRecord, ClassSession


@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = (
        'roll_number',
        'name',
        'registered_on',
        'classes_present',
        'attendance_percentage',
        'face_encoding_status',
    )

    search_fields = ('roll_number', 'name')

    def classes_present(self, obj):
        """Number of classes attended by this student."""
        return AttendanceRecord.objects.filter(
            student=obj
        ).count()

    classes_present.short_description = "Classes Present"

    def attendance_percentage(self, obj):
        """Calculate attendance percentage."""

        # Total number of class sessions actually conducted
        total_classes = ClassSession.objects.count()

        # Classes attended by this student
        present = AttendanceRecord.objects.filter(
            student=obj
        ).count()

        if total_classes == 0:
            return "0%"

        percentage = (present / total_classes) * 100
        percentage = min(percentage, 100)  # safety cap, in case of data inconsistencies

        return f"{percentage:.1f}%"

    attendance_percentage.short_description = "Attendance %"

    def face_encoding_status(self, obj):
        return "Trained" if obj.face_encoding else "Not Trained"

    face_encoding_status.short_description = "Face Encoding"


@admin.register(AttendanceRecord)
class AttendanceRecordAdmin(admin.ModelAdmin):
    list_display = (
        'student',
        'date',
    )

    search_fields = (
        'student__name',
        'student__roll_number',
    )

    list_filter = ('date',)