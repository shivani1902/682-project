from keras.models import Sequential,Model
from keras.layers.core import Flatten, Dense, Dropout
from keras import optimizers
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,rmsprop
import numpy as np
from collections import defaultdict
import operator
from skimage import data, io, filters, transform
import skimage
from itertools import islice
from keras import applications


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def read_tag_list():
	tags = defaultdict(int)
	tags_list = []
	tags_file = open("tag_list.txt")
	only_tags = set()
	lines = tags_file.readlines()
	print lines[10]
	count=0	
	for line in lines:
		#print line
		tags_list.append(line)
 		for i in line.split():
			only_tags.add(i)
#	print sorted_tags
#	only_tags = [x[0] for x in sorted_tags]
	#print only_tags
	#print tags_list
	return list(only_tags), tags_list


def read_image(tags_list,only_tags):
	#print tags_list
	#print tags_list
	# read image path and add them in dictionary {image_path, tags}, got tags from tags_list
	image_tags = {}
	image_file = open("data_list.txt")
	lines = image_file.readlines()
	count = -1
	for line in lines:
		count+=1
		#print count
		#print line, tags_list[count],count
		image_tags[line] = tags_list[count].split()
    	#count = count + 1
    	#
    	print image_tags[line]

#	images = {}
	print len(image_tags)
#	for key,value in image_tags.iteritems():
#		tags_n = [i for i in value if i in only_tags]
		#print only_tags
    #	if len(tags_n)>0:
    #		print key
     #   	images[key] = tags_n
    # PRI
	#n_items = take(100, image_tags.iteritems())
	#print n_items
	return image_tags


def encode_tags(images, n,tags):
	encoded_tags = []
	#print images
	for value in images.values():
		#print value
		temp = np.zeros(n)
    	for i in value:
        	temp[tags.index(i)] = 1
    	encoded_tags.append(temp)
	return encoded_tags

def load_image(path):
    # load image
    img = skimage.io.imread(path)
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
	for key in images.keys():
		if count%10000==0:
			print count
		#print key
		count = count + 1
		im = load_image("../HARRISON/"+key.strip())
    	im -= np.mean(im)
    	image_data.append(im)
	return image_data

def process_data():
	tags, tags_list = read_tag_list()
	images = read_image(tags_list,tags)
#	print images
	encoded_tags = encode_tags(images, len(tags),tags)
	print "done encoding"
	image_data = read_image_data(images)
	print "reading"
	image_data = np.array(image_data).astype(np.float32)
	return image_data, encoded_tags, tags, tags_list


def model(n):
	img_rows, img_cols, img_channel = 224, 224, 3

	base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
	add_model = Sequential()
	add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	add_model.add(Dense(n, activation='sigmoid'))

	model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
	model.compile(loptimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
	return model

def train():
	image_data, encoded_tags, tags, tags_list = process_data()
	#X_train = image_data[:45906]
	#Y_train = encoded_tags[:45906]
	#X_test = image_data[45906:]
	#Y_test = image_data[45906:]
	X_train = image_data[:50]
	Y_train = image_data[:50]

	History = model.fit(np.array(X_train),np.array(Y_train), epochs=5, batch_size=256,validation_split=0.2,shuffle=True )
	with open('/trainHistoryDict', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
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


