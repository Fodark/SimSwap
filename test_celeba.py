import cv2
import torch
import pandas as pd
import numpy as np
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
import os
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet


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


if __name__ == "__main__":
    opt = TestOptions().parse()

    opt.isTrain = False
    opt.use_mask = True
    opt.name = "people"
    opt.Arc_path = "arcface_model/arcface_checkpoint.tar"
    opt.low_res = True
    opt.no_simswap_logo = True

    start_epoch, epoch_iter = 1, 0
    crop_size = 224

    torch.nn.Module.dump_patches = True
    logoclass = watermark_image("./simswaplogo/simswaplogo.png")
    model = create_model(opt)
    model.eval()

    spNorm = SpecificNorm()
    app = Face_detect_crop(name="antelope", root="./insightface_func/models")
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    pairs = pd.read_csv("./data/pairs.csv")
    similarities = []

    with torch.no_grad():

        for row in pairs.itertuples():
            pic_a, pic_b = row
            pic_b_basename = os.path.basename(pic_b)
            # pic_a = opt.pic_a_path)

            img_a_whole = cv2.imread(pic_a)
            if opt.low_res:
                img_a_whole = cv2.resize(img_a_whole, (128, 128))

            img_a_align_crop, _ = app.get(img_a_whole, crop_size)
            img_a_align_crop_pil = Image.fromarray(
                cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB)
            )
            img_a = transformer_Arcface(img_a_align_crop_pil)
            img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
            latend_id = model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)

            # Forward Pass ######################

            # pic_b_folder = opt.pic_b_path
            # pic_b_list = os.listdir(pic_b_folder)

            # pic_b_path = os.path.join(pic_b_folder, pic_b)
            img_b_whole = cv2.imread(pic_b)
            if opt.low_res:
                img_b_whole = cv2.resize(img_b_whole, (128, 128))

            img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
            # detect_results = None
            swap_result_list = []

            b_align_crop_tenor_list = []

            for b_align_crop in img_b_align_crop_list:

                b_align_crop_tenor = _totensor(
                    cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
                )[None, ...].cuda()

                swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tenor_list.append(b_align_crop_tenor)

            if opt.use_mask:
                n_classes = 19
                net = BiSeNet(n_classes=n_classes)
                net.cuda()
                save_pth = os.path.join("./parsing_model/checkpoint", "79999_iter.pth")
                net.load_state_dict(torch.load(save_pth))
                net.eval()
            else:
                net = None

            result = reverse2wholeimage(
                b_align_crop_tenor_list,
                swap_result_list,
                b_mat_list,
                crop_size,
                img_b_whole,
                logoclass,
                os.path.join(opt.output_path, pic_b_basename),
                opt.no_simswaplogo,
                pasring_model=net,
                use_mask=opt.use_mask,
                norm=spNorm,
            )

            # extract the latent vectors in Arcface from img b and result
            img_b_align_crop, _ = app.get(img_b_whole, crop_size)
            img_b_align_crop_pil = Image.fromarray(
                cv2.cvtColor(img_b_align_crop[0], cv2.COLOR_BGR2RGB)
            )
            img_b = transformer_Arcface(img_b_align_crop_pil)
            img_id = img_b.view(-1, img_b.shape[0], img_b.shape[1], img_b.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
            latend_id = model.netArc(img_id_downsample)
            latend_id_b = F.normalize(latend_id, p=2, dim=1)

            result_crop, _ = app.get(result, crop_size)
            result_crop_pil = Image.fromarray(
                cv2.cvtColor(result_crop[0], cv2.COLOR_BGR2RGB)
            )
            result = transformer_Arcface(result_crop_pil)
            img_id = result.view(-1, result.shape[0], result.shape[1], result.shape[2])

            # convert numpy to tensor
            img_id = img_id.cuda()

            # create latent id
            img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
            latend_id = model.netArc(img_id_downsample)
            latend_id_result = F.normalize(latend_id, p=2, dim=1)

            # compute cosine similarity between latent_id_b and latent_id_result
            cos_sim = F.cosine_similarity(latend_id_b, latend_id_result)
            similarities.append(cos_sim.item())

    print(
        "average similarity: ",
        np.mean(similarities),
        "std similarity: ",
        np.std(similarities),
    )
