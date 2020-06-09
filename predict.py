import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import argparse
import json

#image_path="./test_images/"

#1589200286
saved_model="1589185284.h5"

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", help="./test_images/")
parser.add_argument("-m","--model", help="1589185284.h5")
parser.add_argument("-k","--top_k", help="top_k probs of the image")
parser.add_argument("-c","--category_names",help="classes")

args = vars(parser.parse_args())

image_path = args['image']
model = args['model']
top_k = args['top_k']
category_names = args['category_names']

##Load the model
model=tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()

image_size = 224

# Create the process_image function
def process_image(img):
    img = np.squeeze(img)
    image = tf.image.resize(img, (224, 224))   
    image = (image/255)
    return image

# Create the predict function
def predict(image_path,model,top_k=5):  
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    ps = model.predict(image)
    probs, classes= tf.math.top_k(ps, top_k)
    classes += 1
    return probs, classes

if top_k is None:
    top_k=1
probs, classes = predict(image_path, model,top_k=int(top_k))
print('Propabilties:', probs)
print('Classes Keys:', classes)

if category_names != None:
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        print("Classes Values:")
        for idx in classes:
            idx = idx.numpy()
            for i in idx:
                print("-",class_names.get(str(i)))
            

