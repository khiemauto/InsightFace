
import cv2
import asyncio

# from starlette.applications import Starlette
# require jinja2
from starlette.routing import Route, Mount
from starlette.templating import Jinja2Templates
from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
templates = Jinja2Templates(directory="templates")
app = FastAPI()

class Camera:
    def __init__(self,video_source):
        self.video_source = video_source

    async def frames(self):
        video = cv2.VideoCapture(self.video_source)
        if not video.isOpened():
            raise RuntimeError("Could not start video.")

        frame_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

        while True:
            if frame_count == frame_total:
                frame_count = 0
                video = cv2.VideoCapture(self.video_source)
            ret, frame = video.read()
            frame_count += 1

            frame_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
            yield frame_bytes
            await asyncio.sleep(0.01)


async def homepage(request):
    return templates.TemplateResponse("index.html", {"request": request})


async def stream(scope, receive, send):
    message = await receive()
    camera = Camera('rtsp://admin:CongNghe@192.168.1.126:554/Streaming/Channels/101')

    if message["type"] == "http.request":
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"Content-Type", b"multipart/x-mixed-replace; boundary=frame"]
                ],
            }
        )
        while True:
            async for frame in camera.frames():
                data = b"".join(
                    [
                        b"--frame\r\n",
                        b"Content-Type: image/jpeg\r\n\r\n",
                        frame,
                        b"\r\n",
                    ]
                )
                await send(
                    {"type": "http.response.body", "body": data, "more_body": True}
                )







# @app.get("/")
# async def main():
#     return StreamingResponse(Camera(),media_type="image/jpeg")

# routes = [Route("/", endpoint=homepage), Mount("/stream/", stream)]
# app = Starlette(debug=True, routes=routes)
routes = [Route("/", endpoint=homepage), Mount("/stream/", stream)]
app = FastAPI(debug=True, routes=routes)