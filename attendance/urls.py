from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("home/", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("capture/<int:student_id>/", views.capture_page, name="capture_page"),
    path("api/save_face/<int:student_id>/", views.save_face, name="save_face"),
    path('recognize/', views.recognize_page, name='recognize_page'),
    path('mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('report/', views.attendance_report, name='attendance_report'),
    path("teacher/login/", views.teacher_login, name="teacher_login"),
    path("teacher/dashboard/", views.teacher_dashboard, name="teacher_dashboard"),
    path("teacher/start-class/", views.start_class, name="start_class"),
    path("teacher/session/<str:session_code>/", views.view_class_session, name="view_class_session"),
    path("teacher/logout/", views.teacher_logout, name="teacher_logout"),
    path("student/login/", views.student_login, name="student_login"),
    path("student/logout/", views.student_logout, name="student_logout"),
    path("student/dashboard/", views.student_dashboard, name="student_dashboard"),
    path("student/recognize/", views.student_recognize_page, name="student_recognize_page"),
    path("student/<int:student_id>/profile/", views.student_profile, name="student_profile"),
    path('code/', views.enter_session_code, name='enter_session_code'),
    path('teacher/end-session/', views.end_session, name='end_session'),
]