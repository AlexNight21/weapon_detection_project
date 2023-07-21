from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

import urllib.request
import urllib.error
import os
from pathlib import Path

from PIL import Image

import streamlit as st

#st.set_page_config(layout="wide")

cfg = Config.fromfile("./mmdetection/configs/ssd/ssd300_coco.py")
checkpoint = "./model_weights/latest.pth"
cfg.model.bbox_head.num_classes = 1
cfg.model.pretrained = None
cfg.load_from = "./model_weights/latest.pth"
my_model = build_detector(cfg.model)
checkpoint = load_checkpoint(my_model, checkpoint)
my_model.CLASSES = checkpoint['meta']['CLASSES']
my_model.cfg = cfg
my_model.eval()

im_fold = Path("./image")
model_info = Path("./model_weights")

path_config_ssd = "./mmdetection/configs/ssd/ssd300_coco.py"
path_check_def_ssd = "./model_weights/ssd300_coco_20210803_015428-d231a06e.pth"

def get_image_from_url(url):

    destination = f"{im_fold}/Image.jpg"

    rec_url = url

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36')]
    urllib.request.install_opener(opener)

    try:
        local_filename, headers = urllib.request.urlretrieve(url, destination)
    except urllib.error.URLError as e:
        print('Reason: ', e.reason) 

def get_image_from_folder(img):

    image = Image.open(img)
    image.save(f"{im_fold}/Image.jpg")

def show_image():

    placeholder.image(f"{im_fold}/Image.jpg")


st.markdown("<h1 style='text-align: center;'>Weapons detection app</h1>", unsafe_allow_html=True)
st.markdown("---")

st.text("This application allows you to detect weapons in the picture that you will provide")
st.text("You can provide URL-address of image in the field below")

im_url = st.text_input(label = " ", placeholder = "URL-address", label_visibility = "collapsed")

st.text("or download it from your folder")

up_img = st.file_uploader(label = "Choose an image", label_visibility = "collapsed")

if im_url != "":
    get_image_from_url(im_url)
elif up_img is not None:
    get_image_from_folder(up_img)
else:
    pass

_, col2, _ = st.columns([1.5, 1, 1.5])

if os.listdir(im_fold):
    placeholder = st.empty()
    show_image()

but = col2.button("Detect weapons")

if but:

    
    img_path = f"{im_fold}/Image.jpg"

    checkpoint_ssd = path_check_def_ssd
    model_ssd = init_detector(path_config_ssd, checkpoint_ssd, device='cuda:0')
    result = inference_detector(model_ssd, img_path)
    model_ssd.show_result(img_path, result, out_file=f"{im_fold}/Image1.jpg")

    imgg = mmcv.imread(img_path)
    my_result = inference_detector(my_model, img_path)
    my_model.show_result(img_path, my_result, out_file=f"{im_fold}/Image2.jpg")

    placeholder.empty()

    st.markdown("<p><center>Default SSD</center></p>", unsafe_allow_html=True)
    st.image(f"{im_fold}/Image1.jpg")
    
    st.markdown("<p><center>Pretrained SSD on Weapon Dataset</center></p>", unsafe_allow_html=True)
    st.image(f"{im_fold}/Image2.jpg")

for file in os.listdir(im_fold):
    os.remove(f"{im_fold}/{file}")