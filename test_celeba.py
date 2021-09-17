import os
from tqdm import tqdm
import subprocess

src = '/media/dvl1/SSD_DATA/bigraph-dataset/last-363-plain'
out = '/media/dvl1/SSD_DATA/bigraph-dataset/rebuttal/simswap'

f_o = '/media/dvl1/SSD_DATA/bigraph-dataset/rebuttal/simswap/result_whole_swapsingle.jpg'

os.makedirs(out, exist_ok=True)

for f in tqdm(os.listdir(src)):
	pp = os.path.join(src, f)
	_ = subprocess.call(["python", "test_wholeimage_swapsingle.py", "--isTrain", "false", "--use_mask", "--name", "people", "--Arc_path", "arcface_model/arcface_checkpoint.tar", "--pic_a_path", "/media/dvl1/SSD_DATA/bigraph-dataset/032486.jpg", "--pic_b_path", pp, "--output_path", out])
	os.rename(f_o, os.path.join(out, f))
