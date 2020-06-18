# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:37:33 2020

@author: Dell
"""

from flask import request, redirect, url_for, flash, jsonify
import numpy as np
import tensorflow as tf 
from keras.preprocessing import image

from flask import Flask
app = Flask(__name__)


#@app.route('/')
#def hello():
 #   return "Hello World!"

@app.route('/', methods=['POST'])
def predict():
    picture = request.files.get('file')

 #   sample = request.get_json()
    model = tf.keras.models.load_model(r'my_model2')

    test_image = image.load_img(picture, target_size=(125, 125))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = model.predict(test_image,batch_size=1)
    print(result)
    if result == 0:
        return ("0")
    else:
        return ("1")

    return (result)

if __name__ == '__main__':
    app.run()