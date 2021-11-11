import dlib
import cv2
import numpy as np
import os
import torch

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def compute_kp(path):
    clr = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    img_dlib = np.array(clr[:, :, :], dtype=np.uint8)
    dets = detector(img_dlib, 1)

    for k_it, d in enumerate(dets):
        if k_it != 0:
            continue
        kp = []
        landmarks = predictor(img_dlib, d)

        # f_x, f_y, f_w, f_h = rect_to_bb(d)

        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            # x, y = round((x) / width, 4), round((y) / height, 4)
            kp.append([x, y])
        return torch(kp)


def compute_norm_distance(gt, pred):
    # eye_dis = np.linalg.norm(gt[36] - gt[45])
    # d = np.linalg.norm(pred - gt, axis=1).mean() / eye_dis
    eye_dis = (gt[36] - gt[45]).pow(2).sum(0).sqrt().item()
    # print(eye_dis)
    dist = (gt - pred).pow(2).sum(1).sqrt().mean() / eye_dis
    return dist.item()


def compute_stuff(gt_root, pred_root):
    all_files = os.listdir(gt_root)

    dist = []

    for f in all_files:
        gt_path = os.path.join(gt_root, f)
        pred_path = os.path.join(pred_root, f)

        kp_gt = compute_kp(gt_path)
        kp_pred = compute_kp(pred_path)

        if kp_pred is None or kp_gt is None:
            continue

        dist.append(compute_norm_distance(kp_gt, kp_pred))

    print("mean: ", np.mean(dist), "std: ", np.std(dist))


to_compute = [
    {
        "name": "simswap512",
        "skip": False,
        "gen_path": "/media/hdd_data/dvl1/simswap_output/512",
        "test_set": "/media/hdd_data/dvl1/reduced-celebahq-512",
    },
    {
        "name": "simswap128",
        "skip": False,
        "gen_path": "/media/hdd_data/dvl1/simswap_output/128",
        "test_set": "/media/hdd_data/dvl1/reduced-celebahq-128",
    },
    {
        "name": "simswap64",
        "skip": False,
        "gen_path": "/media/hdd_data/dvl1/simswap_output/64",
        "test_set": "/media/hdd_data/dvl1/reduced-celebahq-64",
    },
]

if __name__ == "__main__":
    for exp in to_compute:
        if not exp["skip"]:
            print(f"Starting {exp['name']}")

            kp_dist = compute_stuff(exp["test_set"], exp["gen_path"])

            print(f"EXP NAME: {exp['name']}")
            print(f"Dist: {kp_dist:.5f}")
            print()
