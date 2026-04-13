import cv2
import numpy as np
import threading
import sys
import time
from flask import Flask, Response
from MvCameraControl_class import *   # MVS SDK


# -------------------------------
# Global vars
# -------------------------------
latest_frames = {}   # {cam_index: frame}
frame_locks = {}     # {cam_index: threading.Lock()}
running = True
app = Flask(__name__)


# -------------------------------
# Camera Grabber (MVS SDK)
# -------------------------------
def grab_camera(cam_index, stDeviceList):
    global latest_frames, running

    print(f"[Cam{cam_index}] Starting grab thread...")

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)

    ret = cam.MV_CC_StartGrabbing()
    stOutFrame = MV_FRAME_OUT()
    img_buff = None

    while running:
        ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret == 0:
            width = stOutFrame.stFrameInfo.nWidth
            height = stOutFrame.stFrameInfo.nHeight

            if img_buff is None:
                img_buff = (c_ubyte * (width * height * 3))()

            # Convert to RGB
            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            stConvertParam.nWidth = width
            stConvertParam.nHeight = height
            stConvertParam.pSrcData = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))
            stConvertParam.nSrcDataLen = stOutFrame.stFrameInfo.nFrameLen
            stConvertParam.enSrcPixelType = stOutFrame.stFrameInfo.enPixelType
            stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
            stConvertParam.pDstBuffer = img_buff
            stConvertParam.nDstBufferSize = width * height * 3
            ret = cam.MV_CC_ConvertPixelType(stConvertParam)

            if ret == 0:
                frame = np.asarray(img_buff).reshape((height, width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # fix tint
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

                with frame_locks[cam_index]:
                    latest_frames[cam_index] = frame.copy()

            cam.MV_CC_FreeImageBuffer(stOutFrame)
        else:
            time.sleep(0.005)

    # Cleanup
    cam.MV_CC_StopGrabbing()
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    print(f"[Cam{cam_index}] Stopped.")


# -------------------------------
# Flask Streaming
# -------------------------------
def generate_stream(cam_index):
    global latest_frames, running
    fps = 20
    frame_interval = 1.0 / fps
    last_time = 0

    while running:
        current_time = time.time()
        if (current_time - last_time) < frame_interval:
            time.sleep(0.001)
            continue
        last_time = current_time

        with frame_locks[cam_index]:
            frame = latest_frames.get(cam_index, None)
            if frame is not None:
                frame = frame.copy()

        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<int:cam_index>')
def video_feed(cam_index):
    if cam_index not in latest_frames:
        return f"No camera {cam_index} available", 404
    return Response(generate_stream(cam_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# -------------------------------
# Callable Method
# -------------------------------
def start_streaming(cam_index, host="0.0.0.0", port=5000):
    """
    Start Flask streaming for a given camera index.
    - cam_index: int, the index of the camera
    - host, port: Flask server host and port
    """
    global latest_frames, frame_locks, running

    # Enumerate devices
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        raise RuntimeError("No cameras found!")

    if cam_index >= deviceList.nDeviceNum:
        raise ValueError(f"Camera index {cam_index} not available. Only {deviceList.nDeviceNum} cameras detected.")

    # Prepare single camera
    stDeviceList = cast(deviceList.pDeviceInfo[cam_index], POINTER(MV_CC_DEVICE_INFO)).contents
    latest_frames[cam_index] = None
    frame_locks[cam_index] = threading.Lock()
    t = threading.Thread(target=grab_camera, args=(cam_index, stDeviceList), daemon=True)
    t.start()

    print(f"Camera {cam_index} stream: http://{host}:{port}/video_feed/{cam_index}")
    app.run(host=host, port=port, threaded=True)


# -------------------------------
# Old main disabled (use start_streaming instead)
# -------------------------------
if __name__ == "__main__":
    print("This script is now a module. Use start_streaming(cam_index) instead.")
