# -*- coding: utf-8 -*-
import face_recognition
import cv2
import os

camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_DUPLEX
face_names = []
face_codings = []
person_list = os.listdir("faces/")

for i in range(len(person_list)):
    person_name = os.listdir("faces/" + "person_" + str(i + 1))
    # print(person_name[0])
    face_img = face_recognition.load_image_file("faces/" + "person_" + str(i + 1) + "/" + person_name[0])
    face_codings.append(face_recognition.face_encodings(face_img)[0])
    face_names.append(person_name[0][:person_name[0].index(".")])

while True:
    success, img = camera.read()
    img_new = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    process_this_frame = True
    if process_this_frame:
        marks = face_recognition.face_locations(img_new)
        codings = face_recognition.face_encodings(img_new, marks)
        for coding in codings:
            result = face_recognition.compare_faces(face_codings, coding,0.4)
            print(result)
            for i in range(len(result)):
                if result[i]:
                    name = face_names[i]
                    break
                if i == len(result)-1:
                    name = "unknown"
                #break
        process_this_frame = not process_this_frame
        for (top, right, bottom, left)in (marks):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.imshow('face', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()

cv2.destroyAllWindows()

