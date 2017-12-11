from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
from keras.optimizers import SGD,rmsprop
import numpy as np
from collections import defaultdict
import operator
from skimage import data, io, filters, transform
import skimage
from itertools import islice
from keras import applications
from pathlib2 import Path
import os.path

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def read_image_tags():
    image=[]
    print("reading tag list and data list")
    tag_count = defaultdict(int)
    alltags = set()
    image_data = {}
    tags_file = open("/home/shivanigupta/HARRISON/tag_list.txt")
    tags = tags_file.readlines()
   # print tags
    image_file = open("/home/shivanigupta/HARRISON/data_list.txt")
    lines = image_file.readlines()
    count = -1
    exceptions = 0
    for line in lines:
        path = os.path.join("/home/shivanigupta/HARRISON/"+line.strip())
        count = count + 1
        if os.path.exists(path):
           # try:
            #    im = skimage.io.imread(path)
            #except:
             #   exceptions += 1
              #  pass
            #else:
            im = skimage.io.imread(path)
            resized_image = load_image(im)
            resized_image -= np.mean(resized_image)
            image.append(resized_image)
            image_data[line] = tags[count].split()
            for i in tags[count].split():
                alltags.add(i)
                tag_count[i] += 1
    print ("image data tags:") 
    print (len(image_data))
    return image_data, list(alltags), image

def encode_tags(images, n,tags):
    encoded_tags = []
    tags = tags[:50]
    for value in images.values():
        temp = np.zeros(n)
        for i in value:
            if i in tags:
                temp[tags.index(i)] = 1
        encoded_tags.append(temp)
    return encoded_tags

def load_image(img):
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224,3))
    return resized_img

def read_image_data(images):
    #read images
    image_data = []
    count=0
    #print images
    for key in images.keys():
        count = count + 1
        #print key       
        im = load_image("/home/shivanigupta/hashpred/json/"+key.strip())
        im -= np.mean(im)
        image_data.append(im)
    print ("loaded images:")
    print (len(image_data))
    return image_data

def process_data():
	images,tags,image_data = read_image_tags()
#	print images
	encoded_tags = encode_tags(images, len(tags),tags)
	print ("done encoding")
	#image_data = read_image_data(images)
	#print ("reading")
	image_data = np.array(image_data).astype(np.float32)
	print (image_data.shape)
	return image_data, encoded_tags, tags, images

def create_model(n):
    print(n)
    img_rows, img_cols, img_channel = 224, 224, 3
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    #print(base_model.outpust_shape[1:])
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    #add_model.add(Dense(256,activation='relu'))
    add_model.add(Dense(n, activation='sigmoid'))
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
    return model

def train():
    image_data, encoded_tags, tags, image_with_tags = process_data()
    #X_train = image_data[:45906]
    #Y_train = encoded_tags[:45906]
    #X_test = image_data[45906:]
    #Y_test = image_data[45906:]
    X_train = image_data[:500]
    Y_train = encoded_tags[:500]
    #X_test = []
    #Y_test = []
    print (len(tags))
    model = create_model(50)
    print(np.array(X_train).shape)
    print(np.array(Y_train).shape)
    model.fit(np.array(X_train),np.array(Y_train), epochs=3, batch_size=16,validation_split=0.2,shuffle=True )
    #with open('/trainHistoryDict', 'wb') as file_pi:
     #   pickle.dump(history.history, file_pi)
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

if __name__ == '__main__':
    # start_time = datetime.now()
    train()


