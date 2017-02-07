# set up Python environment: numpy for numerical routines, and matplotlib for plottin
import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from matplotlib import colors
import os
import glob

cmap = colors.ListedColormap(['black', 'red', 'blue', 'green'])
bounds=[0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, ncolors=4)
counter = [0, 0, 0, 0]


# The caffe module needs to be on the Python path;
import sys
caffe_root = '/home/pi/Programs/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

import caffe

model_folder = '/home/pi/Programs/python-programs/roi2patches/models/bigbatch_2/'

caffe.set_mode_gpu()
model_def = model_folder + '/deploy.prototxt' 
model_weights = model_folder + '/snapshot_iter_8900.caffemodel' 

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

#load mean image file and convert it to a .npy file--------------------------------
blob = caffe.proto.caffe_pb2.BlobProto()
data = open('/home/pi/Programs/python-programs/roi2patches/models/bigbatch_2/mean.binaryproto',"rb").read()
blob.ParseFromString(data)
nparray = caffe.io.blobproto_to_array(blob)
os.remove('/home/pi/Temp/pixel/imgmean.npy')
f = file('/home/pi/Temp/pixel/imgmean.npy',"wb")
np.save(f,nparray)

f.close()


# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu1 = np.load('/home/pi/Temp/pixel/imgmean.npy')
# mu1 = mu1.squeeze()
# mu = mu1.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
mu = mu1.squeeze()
# print 'mean-subtracted values:', zip('BGR', mu)
print 'mean shape: ',mu1.shape
print 'data shape: ',net.blobs['data'].data.shape

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# set the size of the input (we can skip this if we're happy

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
# net.blobs['data'].reshape(50,        # batch size
#                         1,         # 3-channel (BGR) images
#                         28, 28)  # image size is 227x227

#load image

def pred(image, x, y):
    transformed_image = transformer.preprocess('data', image)
    # transformed_image = transformer.set_mean('data', mu)
    # copy the image data into the memory allocated for the net
    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()
    output_prob = output['softmax']  # the output probability vector for the first image in the batch
    counter[output_prob.argmax()] += 1
    # print output
    # print 'predicted class is:', output_prob.argmax() + 1
    # print counter
    return output_prob.argmax() + 1

if __name__ == "__main__":
    
    images = sorted(
        glob.glob(
            "/home/pi/Programs/python-programs/roi2patches/data/test_images/*.tif"))
    


    for image in images:
        image_name = image
        im = io.imread(image_name)
        # im = im[500:600, 500:600]
        im = im[200:800, 200:1000]
        x_shape, y_shape = im[:,:,0].shape
        half_patch_size = 14
        pred_map = np.zeros((x_shape, y_shape), dtype='int')
        for x in range(half_patch_size, x_shape-half_patch_size):
            for y in range(half_patch_size, y_shape-half_patch_size):
                patch = im[x-half_patch_size:x+half_patch_size,
                           y-half_patch_size:y+half_patch_size]
                pred_class = pred(patch, x, y)
                pred_map[x][y] = int(pred_class)
    
        save_dir = '/home/pi/Programs/python-programs/roi2patches/data/heatmaps'
        bname = os.path.basename(image_name)
 
        plt.imshow(im, cmap='gray')
        plt.imshow(pred_map, norm=norm, alpha=0.3)
        plt.savefig(os.path.join(save_dir, bname + 'test_bb.png'))
    
        plt.cla()
        plt.pcolor(pred_map, cmap=cmap, norm=norm)
        plt.savefig(os.path.join(save_dir, bname + 'test_colo_bb.png'))
    
        plt.cla()
        plt.imshow(im, cmap='gray')
        plt.savefig(os.path.join(save_dir, bname + 'ori_bb.png'))

