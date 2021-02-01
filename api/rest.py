import json
import os
from typing import List
from fastapi.param_functions import Body, Form

from torch._C import device

from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils import io_utils

from fastapi import FastAPI, File, UploadFile, status, Query
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse

import share_param
import cv2
from core import support, pushserver
import datetime
from typing import Any
from pydantic import BaseModel
import numpy as np


class FaceRecogAPI(FastAPI):
    def __init__(self, system: FaceRecognitionSystem, folders_path: str, db_folder_path: str, title: str = "FaceRecogAPI") -> None:
        super().__init__(title=title)
        self.system = system
        self.db_folder_path = db_folder_path
        self.title = title
        self.dataset_dir = folders_path
        self.image_type = (".jpg", ".jpeg", ".png", ".bmp")

        @self.get('/')
        async def home():
            """
            Home page
            """
            return HTMLResponse("<h1>Face Recognition API</h1><br/><a href='/docs'>Try api now!</a>", status_code=status.HTTP_200_OK)

        @self.post("/predict")
        async def predict(file: UploadFile = File(...)):
            """
            Detect location and recognition face
            file: image file (".jpg", ".jpeg", ".png", ".bmp")
            """
            ext = os.path.splitext(file.filename)[1]
            if ext.lower() not in self.image_type:
                return PlainTextResponse(f"[NG] {file.filename} invaild photo format. Server only accept {self.image_type}", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            try:
                buf = await file.read()
                image = io_utils.read_image_from_bytes(buf)
            except:
                return PlainTextResponse(f"[NG] {file.filename} photo error", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            share_param.detect_lock.acquire()
            share_param.recog_lock.acquire()
            bboxes, landmarks, photo_ids, similarities = self.system.sdk.recognize_faces(image)
            share_param.recog_lock.release()
            share_param.detect_lock.release()
            names = [self.system.get_user_name(uid) for uid in photo_ids]
            
            data = []
            for bbox, landm, name, simi in zip(bboxes, landmarks, names, similarities):
                data.append({
                    "bbox": bbox.tolist(),
                    "landm": landm.tolist(),
                    "name": name,
                    "score": float(simi)})
            return JSONResponse(data, status_code=status.HTTP_200_OK)

        @self.post("/detect")
        async def detect(file: UploadFile = File(...)):
            """
            Detect location of face
            file: image file (".jpg", ".jpeg", ".png", ".bmp")
            """
            ext = os.path.splitext(file.filename)[1]
            if ext.lower() not in self.image_type:
                return PlainTextResponse(f"[NG] {file.filename} invaild photo format. Server only accept {self.image_type}", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            try:
                buf = await file.read()
                image = io_utils.read_image_from_bytes(buf)
            except:
                return PlainTextResponse(f"[NG] {file.filename} photo error", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            share_param.detect_lock.acquire()
            bboxes, landmarks = self.system.sdk.detect_faces(image)
            share_param.detect_lock.release()

            data = []
            for bbox, landm in zip(bboxes, landmarks):
                data.append({
                    "box": bbox.tolist(),
                    "landm": landm.tolist()})
            return JSONResponse(data, status_code=status.HTTP_200_OK)

        @self.post("/add_recognition_queue")
        async def add_recognition_queue(data: str, faceCropExpandFiles: List[UploadFile] = File(...)):
            """
            Recogni from image (112,112,3)
            file: image file (".jpg", ".jpeg", ".png", ".bmp")
            """
            json_data = json.loads(data)

            content = []

            if "deviceId" not in json_data or "bboxs" not in json_data or "landmarks" not in json_data:
                content.append(f"[NG] Not found deviceId or bboxs or landmarks in json key")
                return JSONResponse(content, status_code=status.HTTP_404_NOT_FOUND)

            content = []
            faceCropExpands = []
            for faceCropExpandFile in faceCropExpandFiles:
                ext = os.path.splitext(faceCropExpandFile.filename)[1]
                if ext.lower() not in self.image_type:
                    content.append(f"[NG] {faceCropExpandFile.filename} invaild photo format. Server only accept {self.image_type}")
                    continue
                try:
                    buf = await faceCropExpandFile.read()
                    faceCropExpand = io_utils.read_image_from_bytes(buf)
                    faceCropExpands.append(faceCropExpand)
                except:
                    content.append(f"[NG] {faceCropExpandFile.filename} photo error")
                else:
                    content.append(f"[OK] {faceCropExpandFile.filename} add to recognition queue")

            while share_param.detect_queue.qsize() > share_param.DETECT_SIZE*share_param.batch_size:
                share_param.detect_queue.get()

            share_param.detect_queue.put(
                [json_data["deviceId"], np.asarray(json_data["bboxs"]), np.asarray(json_data["landmarks"]), faceCropExpands, None])

            return JSONResponse(content, status_code=status.HTTP_200_OK)

        @self.post("/get_descriptor")
        async def get_descriptor(file: UploadFile = File(...)):
            """
            Detect location of face
            file: image file (".jpg", ".jpeg", ".png", ".bmp")
            """
            ext = os.path.splitext(file.filename)[1]
            if ext.lower() not in self.image_type:
                return PlainTextResponse(f"[NG] {file.filename} invaild photo format. Server only accept {self.image_type}", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            try:
                buf = await file.read()
                image = io_utils.read_image_from_bytes(buf)
            except:
                return PlainTextResponse(f"[NG] {file.filename} photo error", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)

            image = cv2.resize(image, (112,112))
            share_param.recog_lock.acquire()
            descriptor = self.system.sdk.get_descriptor(image)
            share_param.recog_lock.release()

            return JSONResponse(descriptor.tolist(), status_code=status.HTTP_200_OK)

        @self.post("/api/add_images_database")
        async def add_images_database(user_name: str, files: List[UploadFile] = File(...)):
            """
            Add new photo database for user_name
            user_name: staffID
            files: list image file (".jpg", ".jpeg", ".png", ".bmp")
            """
            content = []
            for file in files:
                ext = os.path.splitext(file.filename)[1]
                if ext.lower() not in self.image_type:
                    content.append(f"[NG] {file.filename} invaild photo format. Server only accept {self.image_type}")
                    continue

                try:
                    buf = await file.read()
                    image = io_utils.read_image_from_bytes(buf)
                except:
                    content.append(f"[NG] {file.filename} photo error")
                    continue

                share_param.detect_lock.acquire()
                share_param.recog_lock.acquire()
                ret, photo_id, photo_path = self.system.add_photo_by_user_name(image, user_name)
                share_param.recog_lock.release()
                share_param.detect_lock.release()
                if ret:
                    content.append(f"[OK] {file.filename} add to {user_name}:{photo_id},{photo_path}")
                else:
                    content.append(f"[NG] {file.filename} not found face or many faces")

            face_datas = support.get_infor(share_param.GET_FACE_INFO_URL, share_param.GET_FACE_INFO_FILE)
            face_dicts = {}

            for face_data in face_datas:
                face_dicts[face_data["StaffCode"]] = face_data
            share_param.face_infos = face_dicts

            self.system.save_database(self.db_folder_path)
            
            return JSONResponse(content, status_code=status.HTTP_201_CREATED)

        @self.post("/api/delete_image_database")
        async def delete_image_database(photo_id: int):
            """
            Remove a photo from database
            photo_id: photo id in database
            """
            ret = self.system.delete_photo_by_photo_id(photo_id)
            if ret:
                return PlainTextResponse(f"Removed {photo_id} from database", status_code=status.HTTP_200_OK)
            else:
                return PlainTextResponse(f"{photo_id} not found", status_code=status.HTTP_404_NOT_FOUND)

        @self.get("/api/reload_database")
        async def reload_database():
            """
            Reload database
            """
            try:
                share_param.detect_lock.acquire()
                share_param.recog_lock.acquire()
                self.system.create_database_from_folders(self.dataset_dir)
                self.system.save_database(self.db_folder_path)
                self.system.load_database(db_folder_path)
                share_param.recog_lock.release()
                share_param.detect_lock.release()
                return PlainTextResponse(f"Reloaded database", status_code=status.HTTP_201_CREATED)
            except:
                return PlainTextResponse(f"Error while reload database", status_code=status.HTTP_417_EXPECTATION_FAILED)

        @self.get("/api/get_images/{user_name}")
        async def get_images(user_name: str):
            """
            Get list image url of user_name
            user_name: staffID
            """
            files_grabbed = {}
            for photoid, (username, photopath) in self.system.photoid_to_username_photopath.items():
                if username == user_name:
                    files_grabbed[photoid] =  os.path.join(f"{share_param.devconfig['FILESERVER']['host']}:{share_param.devconfig['FILESERVER']['port']}", photopath)
            return JSONResponse(files_grabbed, status_code=status.HTTP_200_OK)

        @self.post("/api/delete_user_name")
        async def delete_user(user_name: str):
            ret, nbphoto = self.system.del_photo_by_user_name(user_name)
            
            face_datas = support.get_infor(share_param.GET_FACE_INFO_URL, share_param.GET_FACE_INFO_FILE)
            face_dicts = {}
            for face_data in face_datas:
                face_dicts[face_data["StaffCode"]] = face_data
            share_param.face_infos = face_dicts

            self.system.save_database(self.db_folder_path)

            if ret:
                return PlainTextResponse(f"Removed {nbphoto} photos of {user_name}", status_code=status.HTTP_200_OK)
            else:
                return PlainTextResponse(f"{user_name} not found", status_code=status.HTTP_404_NOT_FOUND)

        @self.get("/find/{user_name}")
        async def find(user_name: str):
            """
            Check information of user_name
            user_name: staffID
            """
            nbimg = 0
            for photoid, (username, photopath) in self.system.photoid_to_username_photopath.items():
                if username== user_name:
                    nbimg += 1
            if nbimg>0:
                return PlainTextResponse(f"{user_name} has {nbimg} photo", status_code=status.HTTP_200_OK)
            else:
                return PlainTextResponse(f"{user_name} not found", status_code=status.HTTP_404_NOT_FOUND)
        
        @self.get("/save_database_to_disk")
        async def save_database_to_disk():
            """
            Save database from RAM to file
            """
            self.system.save_database(self.db_folder_path)
            return PlainTextResponse("[OK] Saved database to disk", status_code=status.HTTP_200_OK)

        @self.get("/update_face_infos")
        async def update_face_infos():
            """
            Update info from web server
            """
            face_datas = support.get_infor(share_param.GET_FACE_INFO_URL, share_param.GET_FACE_INFO_FILE)
            if not face_datas:
                return PlainTextResponse("[NG] Not get info from web server", status_code=status.HTTP_404_NOT_FOUND)

            face_dicts = {}
            for face_data in face_datas:
                face_dicts[face_data["StaffCode"]] = face_data
            share_param.face_infos = face_dicts
            return PlainTextResponse("[OK] success get info from web server", status_code=status.HTTP_200_OK)

        @self.get("/start")
        async def start():
            """
            start server
            """
            share_param.bRunning = True
            return PlainTextResponse("[OK] start server", status_code=status.HTTP_200_OK)

        @self.get("/stop")
        async def stop():
            """
            stop server
            """
            share_param.bRunning = False
            return PlainTextResponse("[OK] stop server", status_code=status.HTTP_200_OK)