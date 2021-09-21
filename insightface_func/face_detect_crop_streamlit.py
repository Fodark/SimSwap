from __future__ import division
import collections
import numpy as np
import glob
import os
import os.path as osp
import cv2
from insightface.model_zoo import model_zoo
from insightface.utils import face_align

__all__ = ["Face_detect_crop", "Face"]

Face = collections.namedtuple(
    "Face",
    [
        "bbox",
        "kps",
        "det_score",
        "embedding",
        "gender",
        "age",
        "embedding_norm",
        "normed_embedding",
        "landmark",
    ],
)

Face.__new__.__defaults__ = (None,) * len(Face._fields)


class Face_detect_crop:
    def __init__(self, name, root="~/.insightface_func/models"):
        self.models = {}
        root = os.path.expanduser(root)
        onnx_files = glob.glob(osp.join(root, name, "*.onnx"))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            if onnx_file.find("_selfgen_") > 0:
                # print('ignore:', onnx_file)
                continue
            model = model_zoo.get_model(onnx_file)
            if model.taskname not in self.models:
                # print("find model:", onnx_file, model.taskname)
                self.models[model.taskname] = model
            else:
                print("duplicated model task type, ignore:", onnx_file, model.taskname)
                del model
        assert "detection" in self.models
        self.det_model = self.models["detection"]

    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        # print("set det-size:", det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname == "detection":
                model.prepare(ctx_id, input_size=det_size)
            else:
                model.prepare(ctx_id)

    def get(self, img, crop_size, anony_mode, max_num=0):
        bboxes, kpss = self.det_model.detect(
            img, threshold=self.det_thresh, max_num=max_num, metric="default"
        )
        if bboxes.shape[0] == 0:  # No faces detected in this frame
            return None

        areas = []
        for bbox in bboxes:
            areas.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        areas = np.array(areas)

        # confidences = bboxes[:, 4]
        max_conf_idx = np.argmax(areas)

        # img = cv2.putText(
        #     img,
        #     f"Biggest face is {str(max_conf_idx)}",
        #     (int(50), int(50)),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA,
        # )

        highest_conf = np.array([bboxes[max_conf_idx]])
        highest_conf_kps = np.array([kpss[max_conf_idx]])

        array_mask = np.ones(bboxes.shape[0], dtype=bool)
        array_mask[max_conf_idx] = 0

        others, others_kp = bboxes[array_mask], kpss[array_mask]

        if anony_mode == "Anonymise you":
            bboxes, kpss = highest_conf, highest_conf_kps
        elif bboxes.shape[0] >= 1:
            bboxes, kpss = others, others_kp
        else:
            return None

        if False:
            for idx, bbox in zip(range(bboxes.shape[0]), bboxes):
                img = cv2.rectangle(
                    img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 0, 255),
                    3,
                )

        align_img_list = []
        M_list = []

        for i in range(bboxes.shape[0]):
            kps = None
            if kpss is not None:
                kps = kpss[i]
            M, _ = face_align.estimate_norm(kps, crop_size, mode="None")
            align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)
            align_img_list.append(align_img)
            M_list.append(M)

        # kps = None
        # if kpss is not None:
        #     kps = kpss[best_index]
        # M, _ = face_align.estimate_norm(kps, crop_size, mode ='None')
        # align_img = cv2.warpAffine(img, M, (crop_size, crop_size), borderValue=0.0)

        return align_img_list, M_list, bboxes
