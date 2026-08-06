from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register_student, name="register_student"),
    path("capture/<int:student_id>/", views.capture_page, name="capture_page"),
    path("api/save_face/<int:student_id>/", views.save_face, name="save_face"),
    path('recognize/', views.recognize_page, name='recognize_page'),
   path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
   path('report/', views.attendance_report, name='attendance_report'),
]