import logging
import logging.handlers
import queue
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import numpy as np
import streamlit as st
import urllib.request
import dlib

detector = dlib.get_frontal_face_detector()

from streamlit_webrtc import (
    VideoProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def app_face_detection():

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class FaceDetector(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
            self.result_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rects = detector(rgb, 1)
            boxes = [convert_and_trim_bb(image, r) for r in rects]
            # loop over the bounding boxes
            for (x, y, w, h) in boxes:
                # draw the bounding box on our image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=FaceDetector,
        async_processing=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence_threshold = confidence_threshold

    if st.checkbox("Show the detected labels", value=True):
        if webrtc_ctx.state.playing:
            labels_placeholder = st.empty()
            # NOTE: The video transformation with object detection and
            # this loop displaying the result labels are running
            # in different threads asynchronously.
            # Then the rendered video frames and the labels displayed here
            # are not strictly synchronized.
            while True:
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )