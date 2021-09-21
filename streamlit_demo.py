import asyncio
import logging
import logging.handlers
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import torch
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
    RTCConfiguration,
)

from video_processor import app_face_detection


from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_streamlit import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from tqdm import tqdm
import fractions
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import time


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer_Arcface = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


opt = TestOptions().parse()
opt.isTrain = False
opt.use_mask = True
opt.name = "people"
opt.Arc_path = "arcface_model/arcface_checkpoint.tar"
opt.no_simswaplogo = True

start_epoch, epoch_iter = 1, 0
crop_size = 224

torch.nn.Module.dump_patches = True

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

name2tensor = {
    "George Clooney": "latents/clooney.pt",
    "Yiming": "latents/yiming.pt",
    "Luca": "latents/luca.pt",
    "Scarlett Johansson": "latents/scarlett.pt",
}


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("Reasearchers' Night demo")

    anony = "Face Anonymisation"

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [anony],
    )
    st.subheader(app_mode)
    if app_mode == anony:
        app_simswap()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_simswap():
    """Simple video loopback"""

    class OpenCVVideoProcessor(VideoProcessorBase):
        def __init__(self) -> None:
            self.condition = "George Clooney"
            self.anony_mode = "selfie"
            # self.demo_image = cv2.imread("./demo_file/group.jpg")
            self.test_mode = False

            self.model = create_model(opt)
            self.model.eval()

            if opt.use_mask:
                n_classes = 19
                self.net = BiSeNet(n_classes=n_classes)
                self.net.cuda()
                save_pth = os.path.join("./parsing_model/checkpoint", "79999_iter.pth")
                self.net.load_state_dict(torch.load(save_pth))
                self.net.eval()
            else:
                self.net = None

            self.spNorm = SpecificNorm()
            self.app = Face_detect_crop(
                name="antelope", root="./insightface_func/models"
            )
            self.app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            self.logoclass = watermark_image("./simswaplogo/simswaplogo.png")
            print("Finished initialization")

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # print("=== START FRAME ===")
            latent_id = torch.load(name2tensor[self.condition], map_location="cuda:0")
            if self.test_mode:
                img = cv2.imread("./demo_file/group.jpg")
            else:
                img = frame.to_ndarray(format="bgr24")

            try:
                img_b_align_crop_list, b_mat_list, _ = self.app.get(
                    img, crop_size, self.anony_mode
                )
            except Exception as e:
                print(f"Error with current frame")
                print(e)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            # detect_results = None
            swap_result_list = []

            b_align_crop_tenor_list = []

            for idx, b_align_crop in enumerate(img_b_align_crop_list):

                b_align_crop_tenor = _totensor(
                    cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
                )[None, ...].cuda()
                # print("\t-- Face", str(idx), "dimensions", _[idx])

                swap_result = self.model(
                    None, b_align_crop_tenor, latent_id, None, True
                )[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            final_img = reverse2wholeimage(
                b_align_crop_tenor_list,
                swap_result_list,
                b_mat_list,
                crop_size,
                img,
                self.logoclass,
                "",
                opt.no_simswaplogo,
                pasring_model=self.net,
                use_mask=opt.use_mask,
                norm=self.spNorm,
                skip_save=True,
            )
            # print("=== END FRAME ===")

            return av.VideoFrame.from_ndarray(final_img, format="bgr24")

    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>",
        unsafe_allow_html=True,
    )

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )

    st.sidebar.image(
        [
            "https://citynews-quicomo.stgy.ovh/~media/original-hi/7125800779587/george-clooney-2-2.jpg",
            "https://my.fbk.eu/fbk-api/v2/picture/ywang?w=250&crop=1",
            "https://my.fbk.eu/fbk-api/v2/picture/lzanella?w=250&crop=1",
            "https://biografieonline.it/img/bio/box/s/Scarlett_Johansson.jpg",
        ],
        # use_column_width="always",
        width=150,
        caption=["George Clooney", "Yiming", "Luca", "Scarlett Johansson"],
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.condition = st.sidebar.radio(
            "Select face to replace with",
            ("George Clooney", "Yiming", "Luca", "Scarlett Johansson"),
        )

        webrtc_ctx.video_processor.anony_mode = st.sidebar.radio(
            "Select anonymisation mode",
            ("Anonymise you", "Selfie mode"),
        )
    else:
        st.sidebar.radio(
            "Select face to replace with",
            ("George Clooney", "Yiming", "Luca", "Scarlett Johansson"),
        )

        st.sidebar.radio(
            "Select anonymisation mode",
            ("Anonymise you", "Selfie mode"),
        )

    # st.markdown(
    #     "This demo is based on "
    #     "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "  # noqa: E501
    #     "Many thanks to the project."
    # )


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(logging.WARNING)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.WARNING)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
