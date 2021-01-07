import os
from typing import List

from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils import io_utils

from fastapi import FastAPI, File, UploadFile, status
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse, StreamingResponse

import share_param


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
            return HTMLResponse("<h1>Face Recognition API</h1><br/><a href='/docs'>Try api now!</a>", status_code=status.HTTP_200_OK)

        @self.post("/predict")
        async def predict(file: UploadFile = File(...)):
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

        @self.post("/api/add_images_database")
        async def add_images_database(user_name: str, files: List[UploadFile] = File(...)):
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
            return JSONResponse(content, status_code=status.HTTP_201_CREATED)

        @self.post("/api/delete_image_database")
        async def delete_image_database(photo_id: int):
            ret = self.system.delete_photo_by_photo_id(photo_id)
            if ret:
                return PlainTextResponse(f"Removed {photo_id} from database", status_code=status.HTTP_200_OK)
            else:
                return PlainTextResponse(f"{photo_id} not found", status_code=status.HTTP_404_NOT_FOUND)

        @self.get("/api/reload_database")
        async def reload_database():
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
            files_grabbed = {}
            for photoid, (username, photopath) in self.system.photoid_to_username_photopath.items():
                if username == user_name:
                    files_grabbed[photoid] =  os.path.join(share_param.devconfig['FILESERVER']['endpoint'], photopath)
            return JSONResponse(files_grabbed, status_code=status.HTTP_200_OK)

        @self.post("/api/delete_user_name")
        async def delete_user(user_name: str):
            ret, nbphoto = self.system.del_photo_by_user_name(user_name)
            if ret:
                return PlainTextResponse(f"Removed {nbphoto} photos of {user_name}", status_code=status.HTTP_200_OK)
            else:
                return PlainTextResponse(f"{user_name} not found", status_code=status.HTTP_404_NOT_FOUND)

        @self.get("/find/{user_name}")
        async def find(user_name: str):
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
            self.system.save_database(self.db_folder_path)
            return PlainTextResponse("[OK] Saved database to disk", status_code=status.HTTP_200_OK)