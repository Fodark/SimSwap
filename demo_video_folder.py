import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_multi import Face_detect_crop
from util.videoswap import video_swap
import os
from tqdm import tqdm
import json


def lcm(a, b):
    return abs(a * b) / fractions.gcd(a, b) if a and b else 0


transformer = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

transformer_Arcface = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


if __name__ == "__main__":
    opt = TestOptions().parse()

    opt.use_mask = True
    opt.isTrain = False
    opt.name = "people"
    opt.Arc_path = "arcface_model/arcface_checkpoint.tar"

    start_epoch, epoch_iter = 1, 0
    crop_size = 224

    torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()

    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    with torch.no_grad():
        pic_a = opt.pic_a_path
        # img_a = Image.open(pic_a).convert('RGB')
        img_a_whole = cv2.imread(pic_a)
        img_a_align_crop, _, _ = app.get(img_a_whole, crop_size)
        img_a_align_crop_pil = Image.fromarray(
            cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
        )
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

        # convert numpy to tensor
        img_id = img_id.cuda()
        # img_att = img_att.cuda()

        # create latent id
        img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)

        legit_videos = []

        for f in tqdm(os.listdir(opt.pic_b_path)):
            input_video = os.path.join(opt.pic_b_path, f)
            output_video = os.path.join(opt.output_path, f)

            is_valid = video_swap(
                input_video,
                latend_id,
                model,
                app,
                output_video,
                temp_results_dir=opt.temp_path,
                no_simswaplogo=True,
                use_mask=opt.use_mask,
            )

            if is_valid:
                legit_videos.append(f)

        json.dump(legit_videos, "valid_videos.json")
