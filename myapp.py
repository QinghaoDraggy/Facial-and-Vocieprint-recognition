from keras.applications.resnet50 import ResNet50
from keras.models import load_model
import os
import wave
import librosa
import numpy as np
import pyaudio
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.engine.network import Network
from tqdm import tqdm
import face_recognition
import cv2
import os
import time


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

layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('resnet.h5')
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

person_feature = []
person_name = []


# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    if len(wav_output) < 8000:
        raise Exception("有效音频小于0.5s")
    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature


# 加载要识别的音频库
def load_audio_db(audio_db_path):
    print('加载音频库')
    audios = os.listdir(audio_db_path)

    pbar=tqdm(total=len(audios))
    for audio in audios:
        path = os.path.join(audio_db_path, audio)
        name = audio[:-4]
        feature = infer(path)[0]
        person_name.append(name)
        person_feature.append(feature)
        pbar.update(1)
        # print("Loaded %s audio." % name)

    print('音频库加载成功')

def recognition(path):
    name = ''
    pro = 0
    feature = infer(path)[0]
    for i, person_f in enumerate(person_feature):
        dist = np.dot(feature, person_f) / (np.linalg.norm(feature) * np.linalg.norm(person_f))
        if dist > pro:
            pro = dist
            name = person_name[i]
    return name, pro

def myapp():
    load_audio_db('audio_db')
    # 录音参数
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "infer_audio.wav"
    #WAVE_OUTPUT_FILENAME = 'C:/Users/asus/Desktop/19-20春夏学期/1A学习/数字信号处理/作业/大作业/VoicePrint/myvoice/黄融杰2.wav'
    print('欢迎进入开锁管家，第一步进行声纹认证')
    #print('经过开锁管家认证，您录音长度为', str(librosa.get_duration(filename=WAVE_OUTPUT_FILENAME)), 'S')

    # 打开录音
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    while True:
        try:
            i = input("按下回车键开机录音，录音3秒中：")
            print("开始录音......")
            flag = True
            frames = []
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                frames.append(data)

            print("录音已结束!")

            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))


            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
    # 识别对比音频库的音频，若通过声纹认证，进行人脸认证
            name, p = recognition(WAVE_OUTPUT_FILENAME)
            if p > 0.7:
                print("识别说话的为：%s，通过声纹认证" % name)
                print("第二步进行人脸认证，请对准摄像头")
                while True:
                    success, img = camera.read()
                    img_new = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

                    process_this_frame = True
                    if process_this_frame:
                        marks = face_recognition.face_locations(img_new)
                        codings = face_recognition.face_encodings(img_new, marks)
                        for coding in codings:
                            result = face_recognition.compare_faces(face_codings, coding, 0.4)
                            for i in range(len(result)):
                                if result[i]:
                                    name = face_names[i]
                                    if flag:
                                        print('识别摄像头中的人为：%s, 通过人脸认证' % name)
                                        print('开锁管家已开锁')
                                    flag = False
                                    break
                                if i == len(result) - 1:
                                    if flag:
                                        print('人脸库没有该用户的面部信息，您未通过人脸认证，请重试')
                                    flag = False
                                    name = "unknown"
                            #if flag:
                                #continue
                        process_this_frame = not process_this_frame
                        for (top, right, bottom, left) in (marks):
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
            else:
                print("音频库没有该用户的语音，您未通过声纹认证，请重试")
        except:
            pass

if __name__ == '__main__':
    myapp()
