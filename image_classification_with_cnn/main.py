import os
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import numpy as np


base_width = 224
generated_image_path = os.path.abspath("D:\\Projelerim\\ML\\image_classification_with_cnn\\generated_images")

img_path = os.path.abspath("D:\\Projelerim\\ML\\image_classification_with_cnn\\images")
files = [img_path + '\\' + x for x in os.listdir(img_path) if 'jpg' in x]
resized_img_path = os.path.abspath("D:\\Projelerim\\ML\\image_classification_with_cnn\\resized_images")
resized_files = [resized_img_path + '\\' + x for x in os.listdir(resized_img_path) if 'jpg' in x]
generated_files = [generated_image_path + '\\' + x for x in os.listdir(generated_image_path) if 'jpeg' in x]
