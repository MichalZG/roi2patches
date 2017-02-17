from labels import Image, Label, Roi, Mask
from patches import Patch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import shutil 
import os
import h5py
import time

train_part = 0.5

patch_shape = [5, 5, 3]
chunk_size = 100

images_path = '/home/pi/Programs/python-programs/roi2patches/data/source_images/test_data/'
db_dir = '/home/pi/Programs/python-programs/roi2patches/data/datasets'
db_name = '71x71_rgbtest.h5'


def create_dsets_dict(images_path, im_ext='*.tif', train_part=0.8):

    images = sorted(glob.glob(os.path.join(images_path, im_ext)))
    images = [image for image in images if 'label' not in image]
    random.shuffle(images)
    train_pack, val_pack = images[:int(train_part*len(images))], \
            images[int(train_part*len(images)):]
    #test_pack, val_pack = images[:4], images[4:]

    dsets_dict = {'X': train_pack, 'X_val': val_pack}
    
    return dsets_dict


def create_db(db_dir, db_name, patch_shape,
              chunk_size, compression=None):
    
    max_shape = [None] + patch_shape
    dset_shape = [0] + patch_shape
    chunk_shape = tuple([chunk_size] + patch_shape)
    db_path = os.path.join(db_dir, db_name)

    db = h5py.File(db_path, 'w')
    db.create_dataset('X', dset_shape, maxshape=max_shape,
                       chunks=chunk_shape, compression=compression)
    db.create_dataset('Y', (0,), maxshape=(None,),
                       chunks=(chunk_size,), compression=compression)
    db.create_dataset('X_val', dset_shape, maxshape=max_shape,
                       chunks=chunk_shape, compression=compression)
    db.create_dataset('Y_val', (0,), maxshape=(None,),
                       chunks=(chunk_size,), compression=compression)
    
    db.close()


def open_db(db_dir, db_name, attr):
    
    db_path = os.path.join(db_dir, db_name)
    db = h5py.File(db_path, attr)

    return db


def add_to_dset(db, dset_type, arr, label):
    dset = db[dset_type]
    dset_len = dset.shape[0]
    dset.resize([dset_len+1] + patch_shape)
    dset[dset_len-1:dset_len] = arr
    
    dset_type = dset_type.replace('X', 'Y')
    dset = db[dset_type]
    dset_len = dset.shape[0]
    dset.resize([dset_len+1])
    dset[dset_len-1:dset_len] = int(label)
    

def create_mean(db, dsets_list):
    mean_dict = {}
    dsets_list = [ds for ds in dsets_list if 'Y' not in ds]
    for dset_name in dsets_list:
        dset = db[dset_name]
        mean_arr = np.zeros(patch_shape, dtype=np.float32)
        chunk_size = dset.chunks[0]
        dset_len = dset.shape[0]
        for i in range(0, dset_len, chunk_size):
            mean_arr += np.sum(dset[i:i+chunk_size], axis=0) / dset_len
        
        save_mean_image(db, mean_arr, dset_name)
        mean_dict[dset_name] = mean_arr

    return mean_dict


def save_mean_image(db, mean_arr, dset_name):
    save_dir = db.filename + dset_name
    np.save(save_dir, mean_arr)


def mean_substract(db, mean_dict):
    for dset_name in mean_dict.keys():
        dset = db[dset_name]
        chunk_size = dset.chunks[0]
        dset_len = dset.shape[0]
        mean_arr = mean_dict[dset_name]
        for i in range(0, dset_len, chunk_size):
            dset[i:i+chunk_size] -= mean_arr


def make_patches(db, images_pack, dset_type, patch_shape):

    start = time.time()
    counter = [0, 0, 0, 0]
    n_counter = 0
    for image in images_pack:
        print(image)
        im = Image(image)
        im.create_roi()
        label = Label(im)
        label.save_label_image()
        mask = Mask(im)
        mask.load_rel_label(label)
        mask.get_data()
        shape = mask.data.shape
        px, py, pz = patch_shape
        mask_arr = mask.data[px:shape[0]-px,
                             py:shape[1]-py]
        for (x,y), value in np.ndenumerate(mask_arr):
            x += px
            y += py
            flag = value

            if flag != 0:
                if (flag == 3 and n_counter < 2):
                    n_counter += 1
                else:
                    counter[flag-1] += 1
                    n_counter = 0
                    p = Patch(mask, [px, py], [x, y], '', '')
                    p.set_type(flag)
                    p.create_data()
                    arr = p.data
                    label = flag - 1
                    add_to_dset(db, dset_type, arr, label)

    print(counter)
    counter = [0, 0, 0, 0]
    print(time.time() - start)


if __name__ == "__main__":

    print('create db')
    create_db(db_dir, db_name, patch_shape,
              chunk_size, compression='lzf')
    db = open_db(db_dir, db_name, 'a')
    dsets_dict = create_dsets_dict(images_path, train_part=train_part)
    print('pathes create')
    for dset_name, dset_images in dsets_dict.items():
        make_patches(db, dset_images, dset_name, patch_shape)
    db.close() 
    db = open_db(db_dir, db_name, 'r')
    mean_dict = create_mean(db, dsets_dict.keys())
    db.close()
    db = open_db(db_dir, db_name, 'r+')
    print('mean sebstract')
    mean_substract(db, mean_dict)
    db.close()
