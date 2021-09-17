import argparse
import pandas as pd
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str, required=True)
parser.add_argument("--video", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.csv_file, names=['frame_index', 'face_index', 'x1', 'y1', 'x2', 'y2'])
cap = cv2.VideoCapture(args.video)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

def check(value):
    if value < 0:
        return 0
    return value

frame_index = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        faces = df[df.frame_index == frame_index].values.tolist()

        # print(f"Frame index: {frame_index}")
        # print(faces.values.tolist())
        for face in faces:
            _, face_index, x1, y1, x2, y2 = face
            # print(x1, x2, y1, y2)
            x1, y1, x2, y2 = check(x1), check(y1), check(x2), check(y2)
            # print(x1, y1, x2, y2)
            extracted_face = frame[y1:y2, x1:x2]

            cv2.imwrite(f"/media/dvl1/SSD_DATA/wildtrack-dataset/quantitative_test/gen/{frame_index}_{face_index}.png", extracted_face)
            # print(f"{frame_index}: face {face_index}")
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"Frame: {frame_index}")
        # exit(0)
        # cv2.imshow('Frame',frame)
    
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     break

    
    else:
        break

cap.release()

