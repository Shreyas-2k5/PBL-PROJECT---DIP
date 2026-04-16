import cv2

cascade = cv2.CascadeClassifier("app/models/haarcascade_frontalface_default.xml")

print("Loaded:", not cascade.empty())