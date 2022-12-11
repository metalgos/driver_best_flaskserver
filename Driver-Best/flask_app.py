#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 환경 셋팅
import h5py
# import necessary packages
import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

import cv2, dlib  # 이미지 표현 및 눈 좌표 얻어오기
import numpy as np  # 데이터 처리
from imutils import face_utils  # 얼굴 분석
from tensorflow import keras  # 모델 학습 및 테스트
#from playsound import playsound  # 소리 재생
#import threading  # 스레드 사용




import os # 파일 마지막꺼 출력
#import asyncio#비동기 함수 사용
import requests#통신 관련


#파일 업로드 처리,플라스크 관련
from flask import Flask, render_template, request, redirect, url_for
from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
import urllib.request

# 추출할 눈 이미지 사이즈
IMG_SIZE = (34, 26)
# 눈을 감은 프레임을 세줄 변수
#n_count = 0
# 경보음 다중재생 방지 변수
global is_playing
is_playing = False

# 얼굴에 68개의 점을 찍음
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/kyscodeman/mysite/shape_predictor_68_face_landmarks.dat')

# 학습한 모델을 불러옴

#졸음운전
model = keras.models.load_model('/home/kyscodeman/mysite/models/2018_12_17_22_58_35.h5') #원본소스
#model = keras.models.load_model('models/2022_12_04_04_04_59.h5')
#model.summary()

#도난방지
car_500 = load_model('/home/kyscodeman/mysite/theaf_modelh5/car_500.h5')
car_500.summary()


#절대경로찾기
from pathlib import Path

# 눈을 찾아주는 함수
def crop_eye(gray, eye_points):

    #IMG_SIZE = (34, 26)

    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)
    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# def playSound():
#     global is_playing
#     t = threading.Thread(target=thread)
#     # 스레드를 시작한후 join해서 동작이 멈추길 기다린 후 is_plaing을 false로 바꿔 소리 재생이 가능하도록.
#     t.start()
#     t.join()
#     is_playing = False

# def thread():
#     playsound("/home/kyscodeman/mysite/sound.wav")

def vid_to_img(): # 비디오 추출해어 이미지 만들기. 사용할지 말지 결정
    fps2 = int(cap.get(cv2.CAP_PROP_FPS))

    count = 0
    img_count=1
    while(cap.isOpened()):

        ret, image = cap.read()

        #if(int(cap.get(1)) % fps2 == 0): #앞서 불러온 fps 값을 사용하여  초마다 추출
        try:
            if(img_count%5==0): # 5프레임으로
                #cv2.imwrite(incode_path[:-4] + "/frame%d.jpg" % count, image)
                cv2.imwrite(incode_img+"//"+str(count)+".jpg", image)
                count += 1
            img_count+=1
        except:
            break
    cap.release()

# def vid_path():  #dir 위치 읽어와서 비디오 죄다 읽은후 마지막 비디오 반환하기 , 사용안함
#     global file_path
#     global cap
#     cap = cv2.VideoCapture(file_path)
#     for (root, directories, files) in os.walk(dir_path): #dir_path 내의 파일중 마지막 파일을
#         for file in files:
#             file_path = os.path.join(root, file) # file_path에 값 저장


#플라스크 실행
from flask import Flask #간단히 플라스크 서버를 만든다

app = Flask(__name__)

from flask import Flask

app = Flask(__name__)

@app.route("/testsleepy") #테스트용 주소
def gotojsp2():
    URL = "http://driverbest.net/file/sleepystate?deepresult=65"
    response = requests.get(URL)
    return "TEST SLEEPY. GOTO JSP sleepy on"


@app.route("/testtheaf") #테스트용 주소
def gotojsp3():
    URL = "http://driverbest.net/file/theafstate?deepresult=65"
    response = requests.get(URL)
    return "TEST theaf. GOTO JSP sleepy on"


def gotojsp(result): #졸음 처리 결과 jsp페이지 반환
    URL = "http://driverbest.net/file/sleepystate?deepresult="+result
    response = requests.get(URL)

def gotojsp_theaf(result): #졸음 처리 결과 jsp페이지 반환
    URL = "http://driverbest.net/file/theafstate?deepresult="+result
    response = requests.get(URL)


#플라스크 실행시 접속할 주소, 딥러닝 졸음운전 자동 실행
@app.route("/sleepy")
def sleepy(login_no,new_dir):

    n_count=0

    cap=cv2.VideoCapture(new_dir+"test_vid.mp4")

    new_count = 0
    new_fps=5
    while cap.isOpened():


        new_count+=1
        ret, img_ori = cap.read()  #ret = 프레임 읽기을 성공하면 True 값 반환 #img_ori = 배열 형식의 영상 프레임 (가로 X 세로 X 3) 값 반환

        if not ret: #ret 이 false 면 탈출
            break


        if(new_count%new_fps==0): #프레임 5개당 하나만 분석하기. 속도를 위해

            img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
            #img_ori = cv2.resize(img_ori, dsize=(320, 240))

            img = img_ori.copy()
            #global gray
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



            faces = detector(gray)

            for face in faces:

                shapes = predictor(gray, face)
                shapes = face_utils.shape_to_np(shapes)

                eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
                eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

                eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
                eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
                eye_img_r = cv2.flip(eye_img_r, flipCode=1)

                # cv2.imshow('l', eye_img_l)
                # cv2.imshow('r', eye_img_r)

                eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
                eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

                pred_l = model.predict(eye_input_l)
                pred_r = model.predict(eye_input_r)

                if pred_l < 0.1 and pred_r < 0.1:
                    n_count += 1
                else:
                    n_count = 0
                    is_playing = False
                    # playsound("")

                    #프레임수 조절, 30프레임은 1초
                if n_count > 20:
                    cv2.putText(img, "Wake up", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # 스레드에서 사운드 재생
    #                 if not is_playing:
    #                     is_playing = True
    #                     t = threading.Thread(target=playSound)
    #                     t.start()
                    gotojsp(login_no)      #판정결과 졸음일경우 jsp 접속 함수 호출
                    n_count=0  #판정위해 추가,. 졸음 판정되면 즉시 초기화
                    print("졸음")

                # visualize
                state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
                state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

                state_l = state_l % pred_l
                state_r = state_r % pred_r

                # 색 지정
                if pred_l > 0.1:
                    l_color = (255, 255, 255)
                else:
                    l_color = (0, 0, 255)
                if pred_r > 0.1:
                    r_color = (255, 255, 255)
                else:
                    r_color = (0, 0, 255)

                cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(l_color), thickness=2)
                cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(r_color), thickness=2)

                cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (l_color), 2)
                cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (r_color), 2)

                #print("판독 실행중")


#                     img = cv2.resize(img, dsize=(640, 480))
#                     cv2.imshow('result', img)
#                     if cv2.waitKey(1) == ord('q'):
#                         cv2.destroyAllWindows()
#                         break


    cap.release()

    return "test_sleepy"


@app.route("/theaf")
def theaf(login_no,new_dir):



    n_count=0 #도난방지 확인 프레임 버퍼


    # open webcam
    webcam = cv2.VideoCapture(new_dir+"test_vid.mp4")

    new_count = 0
    new_fps=10


    # loop through frames
    while webcam.isOpened():

        new_count+=1
        # read frame from webcam
        status, frame = webcam.read()

        if not status:
            print("Could not read frame")
            exit()

            # apply face detection


        if(new_count%new_fps==0):
            # apply face detection
            face, confidence = cv.detect_face(frame)
            # loop through detected faces
            for idx, f in enumerate(face):

                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:

                    face_region = frame[startY:endY, startX:endX]

                    face_region1 = cv2.resize(face_region, (224, 224), interpolation = cv2.INTER_AREA)

                    x = img_to_array(face_region1)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)

                    prediction = car_500.predict(x)

                    if prediction < 0.7: # 마스크 미착용으로 판별되면,
                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "driver {:.2f}".format((1 - prediction[0][0]))
                        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        n_count = 0

                    else: # 마스크 착용으로 판별되면

                        n_count += 1

                        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                        Y = startY - 10 if startY - 10 > 10 else startY + 10
                        text = "other {:.2f}".format(prediction[0][0])
                        cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    if n_count > 5:
                        print("도난")

                        gotojsp_theaf(login_no)
                        n_count=0

#             # display output
#             cv2.imshow("driver other classify", frame)

#             # press "Q" to stop
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

        #release resources
    webcam.release()
    cv2.destroyAllWindows()


    return "theaf deppruning page"



def id_split(f_text): # 입력 영상 이름 나눠서 배열로 저장
    strings = f_text.split("_")
    return strings[1] #세션 로그인 아이디넘버 반환



@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():

    dir_path2 = "/home/kyscodeman/mysite/checkc_upload_vid/"

    empty_list=[]
    empty_list_length = len(empty_list)
    if request.method == 'POST':
        f = request.files['file']
        login_no = id_split(f.filename) #파일이름 뽑아서 세션넘버값 가져오기

        new_dir = dir_path2+"/"+login_no+"/"
        if(os.path.isdir(new_dir)!=True): #폴더 경로 체크 . 없으면 생성
            os.mkdir(new_dir)

        # 저장할 경로 + 파일명
        f.save(new_dir+secure_filename("test_vid.mp4"))
        #f.filename
        login_no = id_split(f.filename)

    sleepy(login_no,new_dir)

    return "upload_test_vid"

@app.route('/fileUpload_theaf', methods = ['GET', 'POST'])
def upload_file_theaf():

    dir_path2 = "/home/kyscodeman/mysite/checkc_upload_vid_thef/"

    empty_list=[]
    empty_list_length = len(empty_list)
    if request.method == 'POST':
        f = request.files['file']
        login_no = id_split(f.filename) #파일이름 뽑아서 세션넘버값 가져오기

        new_dir = dir_path2+"/"+login_no+"/"
        if(os.path.isdir(new_dir)!=True): #폴더 경로 체크 . 없으면 생성
            os.mkdir(new_dir)




        # 저장할 경로 + 파일명
        f.save(new_dir+secure_filename("test_vid.mp4"))
        #f.filename
        login_no = id_split(f.filename)
    theaf(login_no,new_dir)

    return "upload_test_vid_thef"


@app.route('/', methods = ['GET', 'POST'])
def hello():
    return "welcome kyscodeman flask site!"


# if __name__ == '__main__':
#     app.run(debug=False,host="127.0.0.1",port=5000)

# if __name__ == '__main__':
#     if 'liveconsole' not in gethostname():
#         app.run(debug=True)

# In[ ]:



