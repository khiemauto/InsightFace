import os
import shutil
from distutils.command.upload import upload
from fileinput import filename
from io import BytesIO
from os.path import join as joinpath
from random import random
from typing import List

import cv2
import numpy as np
from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils.draw_utils import draw_boxes, draw_landmarks
from face_recognition_sdk.utils.io_utils import (read_image, read_yaml,
                                                 save_image)
from fastapi import FastAPI, File, Response, UploadFile, status
from matplotlib import use
from PIL import Image
from requests import request
from starlette.requests import Request
from starlette.responses import StreamingResponse

import share_param


class FaceRecogAPI(FastAPI):
    def __init__(self, system: FaceRecognitionSystem, folders_path: str, db_folder_path: str, title: str = "FaceRecogAPI") -> None:
        super().__init__(title=title)
        self.system = system
        self.db_folder_path = db_folder_path
        self.title = title
        self.dataset_dir = folders_path

        @self.get('/')
        async def home():
            return "Face Recognition API"

        @self.post("/predict/image", status_code=status.HTTP_200_OK)
        async def predict(request: Request, threshold: float = 0.5, file: bytes = File(...)):
            data = {"success": False}
            if request.method == "POST":
                data = self.recognition_face(file, threshold)
                print(data)
            return data

        @self.post("/detect/image", status_code=status.HTTP_200_OK)
        async def predict(request: Request, file: bytes = File(...)):
            try:
                data = self.detect_face(file)
            except:
                data = {"success": False}
            print(data)
            return data

        @self.post("/api/add/image", status_code=status.HTTP_201_CREATED)
        async def create_upload_file(user_name, uploaded_file: UploadFile = File(...)):
            if user_name not in os.listdir(self.dataset_dir):
                return {f"status': 'User id:{user_name} not found"}
            else:
                file_location = joinpath(
                    self.dataset_dir, user_name, uploaded_file.filename)
                print(file_location)
                with open(file_location, "wb+") as file_object:
                    # print(len(uploaded_file.file))
                    shutil.copyfileobj(uploaded_file.file, file_object)
                return {"info": f"file '{uploaded_file.filename}' saved at '{file_location}'"}

        @self.post("/api/add/image/database", status_code=status.HTTP_201_CREATED)
        async def add_database(user_name, uploaded_file: UploadFile = File(...)):
            """ Can only add 1 image at a time"""
            if user_name not in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} not found".format(user_name)}
            else:
                file_location = joinpath(
                    self.dataset_dir, user_name, uploaded_file.filename)
                with open(file_location, "wb+") as file_object:
                    shutil.copyfileobj(uploaded_file.file, file_object)
                image = read_image(file_location)

                user_id = [
                    id for id, name in self.system.user_id_to_name.items() if name == user_name][0]
                print(user_id)
                self.system.sdk.add_photo_by_user_id(image, user_id)
                return {
                    "status": uploaded_file.filename,
                    "add to": user_name
                }

        @self.get("/api/database/reload")
        async def reset_database():
            try:
                share_param.detect_lock.acquire()
                share_param.recog_lock.acquire()
                self.system.create_database_from_folders(self.dataset_dir)
                self.system.save_database(self.db_folder_path)
                self.system.load_database(db_folder_path)
                share_param.recog_lock.release()
                share_param.detect_lock.release()
                return {"create new database"}
            except Exception as e:
                print(e)
                return {e}

        @self.get("api/get/{user_id}/image")
        async def get_image(user_id):
            if user_id not in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} not found".format(user_id)}
            else:
                cv2img = self.get_random_image(user_id)
                res, im_png = cv2.imencode(".png", cv2img)
                return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/png")

        @self.post("/api/add/{user_id}/name")
        async def add_user(user_id):
            if user_id in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} already exists".format(user_id)}
            else:
                os.chdir(self.dataset_dir)
                os.makedirs(user_id)
                # self.system.sdk.add_descriptor(user_id)
                return {"folder create {}".format(user_id)}

        @self.post("/api/delete/{user_id}/folder")
        async def delete_user(user_id):
            if user_id not in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} not found".format(user_id)}
            else:
                os.chdir(self.dataset_dir)
                os.removedirs(user_id)
                return{
                    "folder {} remove".format(user_id)
                }

        @self.post("api/delete/{user_id}/database")
        async def delete_user_database(user_id):
            if user_id not in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} not found".format(user_id)}
            else:
                self.system.sdk.delete_user_by_id(user_id)
                return{
                    "user id {} delete from database".format(user_id)
                }

        @self.get("/find/{user_id}")
        async def find(user_id):
            if user_id not in os.listdir(self.dataset_dir):
                return{"status': 'User id:{} not found".format(user_id)}
            else:
                return {
                    "User_id: {}".format(user_id)
                }

        # @self.get("find/database/{user_id}")
        # async def find_database_user(user_id):
        #     if user_id in self.get_list_of_id():
        #         return{True}
        #     else:
        #         return{False}

    def buffer_to_image(self, imgbuffer):
        image = np.frombuffer(imgbuffer, dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def get_random_image(self, user_id):
        dirpath = joinpath(self.dataset_dir, user_id)
        dir = os.listdir(dirpath)
        return random.choice(dir)

    def recognition_face(self, binaryimg, threshold):
        data = {"success": False}
        if binaryimg is None:
            return data
        image = self.buffer_to_image(binaryimg)
        bboxes, landmarks, user_ids, similarities = self.system.sdk.recognize_faces(
            image)
        names = [self.system.get_user_name(uid) for uid in user_ids]

        names_score = {}

        for name, score in zip(names, similarities):
            if score > threshold:
                names_score[name] = float(score)
        return names_score

    def get_list_of_id(self):
        file_path = joinpath(self.db_folder_path,"id_to_username.pkl")
        import pickle
        with open(file_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        return data

    def detect_face(self, imgbuffer):
        data = {"success": False}
        if not imgbuffer:
            return data
        preprocess_img = self.buffer_to_image(imgbuffer)
        boxes, landms = self.system.sdk.detect_faces(preprocess_img)

        list_boxlandmdicts = []

        for box, landm in zip(boxes, landms):
            boxlandmdict = {}
            boxlandmdict["box"] = box.tolist()
            boxlandmdict["landm"] = landm.tolist()
            list_boxlandmdicts.append(boxlandmdict)
        return list_boxlandmdicts

    def save_image(self, binaryimg, image_path):
        data = {"success": False}
        if binaryimg is None:
            return data
        image = self.buffer_to_image(binaryimg)
        output_image = Image.fromarray(image, "RGB")
        output_image.save(image_path)
