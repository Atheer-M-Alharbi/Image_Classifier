import argparse
import json
import tensorflow as tf  
import tensorflow_hub as hub
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Predict the top flower names from an image along with their corresponding probabilities') 
parser.add_argument('image', help=' Path of the image to predict')
parser.add_argument('model', default='my_model.h5', help='Saved model path',type=str)
parser.add_argument('--top_k', default=5, help=' The number of top_k result',type=int)
parser.add_argument('--category_names', default='label_map.json', help='category namees')

args, unknown = parser.parse_known_args() 

loaded_model = tf.keras.models.load_model(args.model,custom_objects={'KerasLayer':hub.KerasLayer})

with open(args.category_names , 'r') as f:
    class_names = json.load(f)
    
def process_image(image): 
    TensorImage = tf.convert_to_tensor(image)
    ResizeImage = tf.image.resize(TensorImage, (224,224))
    normlizedImage = tf.cast(ResizeImage, tf.float32)
    normlizedImage /= 255
    return normlizedImage.numpy()
    
def predict(image, loaded_model, top_k): 
    image = Image.open(image)
    test_image = np.asarray(image)
    processed_image = process_image(test_image)
    newimage = np.expand_dims(processed_image, axis =0)
    probabilities = loaded_model.predict(newimage)
    probabilities = probabilities.tolist()
    values, indices = tf.math.top_k(probabilities , k=top_k)
    probs = values.numpy().tolist()[0]
    classes = indices.numpy().tolist()[0]
    flower_names = []
    for int_lable in classes:
         flower_names.append(class_names[str(int_lable+1)])
    print('Top flower names: ',flower_names,'\n Probabilities',probs)


if __name__ == "__main__":
     predict(args.image, loaded_model , args.top_k)