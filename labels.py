import numpy as np
import os
from read_roi import read_roi_zip
import cv2

class Image:
    """
    Image class
    """
    def __init__(self, im_dir, rois_dir=None, label_dir=None, mask_dir=None):
        self.dir = im_dir
        self.name = os.path.basename(im_dir).replace('.tif', '') 
        self.data = self.load_data()
        self.tumor_type = None
        self.roi = None
        self.rois_dir = rois_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.label = None
        self.mask = None

    
    def set_tumor_type(self, tumor_type):
        """
        Set tumor type
        Args:
            tumor_type (str): must be benign, fibroadenoma or malignant \
                    raise value error otherwise 
        """
        tumor_types_list = ['benign', 'fibroadenoma', 'malignant']
        tumor_type = tumor_type.lower()

        if tumor_type in tumor_types_list:
            self.tumor_type = tumor_type
        else:
            raise ValueError('Wrong tumor type')


    def load_data(self):
        data = cv2.imread(self.dir)
        return data


    def get_data(self):
        """
        Return image data as array, need cv2
        """
        if self.data:
            return self.data
        else:
            self.load_data()


    def get_rois(self):
        """
        Return dict of roi objects, if roi does not exist return None
        """
        if self.roi:
            return self.roi.get_rois()
        else:
            roi = Roi(self, self.rois_dir)
            self.roi = roi
            return roi.get_rois()


    def get_label(self):
        """
        Return label object if label does not exist return None
        """
        if self.label:
            return self.label
        else:
            label = Label(self, self.label_dir)
            return label


    def get_mask(self, mask_dir):
        """
        Return maks object, if mask does not exist return None
        """
        if self.mask:
            return self.mask
        else:
            mask = Mask(self, self.mask_dir)
            return mask


    def create_roi(self):
        self.roi = Roi(self, self.rois_dir)


class Roi:
    """
    Roi class
    Args:
        rel_image_dir (str): dir to related image
    """
    def __init__(self, rel_image, rois_dir=None):
        self.rel_image = rel_image
        self.dir = self.create_dir(rois_dir)
        self.nuclei, self.cytoplasm, self.background = [None] * 3
        self.rois = {}
        self.create_rois()


    def create_dir(self, rois_dir):
        """
        create roi dir
        """
        if rois_dir is not None:
            self.dir = rois_dir
        else:
            head, _ = os.path.split(self.rel_image.dir)
            self.dir = head


    def get_rois(self):
        """
        Return dict of rois
        """
        if self.rois:
            return self.rois
        else:
            if (self.nuclei and self.cytoplasm and self.background) is not None:
                pass
            else:
                self.create_rois()

            self.rois = {'nuclei': self.nuclei,
                         'cytoplasm': self.cytoplasm,
                         'background': self.background}

            return self.rois


    def create_rois(self):
        """
        Create rois arrays with format for cv2
        """
        head, tail = os.path.split(self.rel_image.dir)
        name_prefix = tail.replace('.tif', '')
        name_suffix = '.tif.zip'
        for _type in ['nuclei', 'cytoplasm', 'background']:
            xy_list = []

            try:
                roi_array = read_roi_zip(os.path.join(
                    head, ''.join([name_prefix, '_', _type, name_suffix])))
            except FileNotFoundError:
                print('{} roi not found'.format(_type))

            for r in roi_array.values(): # FIXME
                xy = []
                for x, y in zip(r['x'], r['y']):
                    xy.append([x, y])
                xy_list.append(np.array(xy))
            setattr(self, _type, xy_list)

        self.get_rois()


class Label:
    """
    Label class, contain masks created from rgb images bases on fuji roi

    Args:
        rel_image (str): realted image object
    """
    def __init__(self, rel_image, update=True,
                 rois_dir=None, label_dir=None):
        self.rel_image = rel_image
        self.dir = self.create_dir(label_dir)
        self.roi = Roi(rel_image, rois_dir)
        self.exist = self.check_exist()
        self.update = update
        self.data = None
        self.ds = {'background': (-1, (255, 0, 0), -1), # FIXME
                   'cytoplasm': (-1, (0, 255, 0), -1),
                   'nuclei': (-1, (0, 0, 255), -1),
                   'border': (-1, (0, 0, 0), 2)}


    def create_dir(self, label_dir):
        """
        Create label dir
        """
        if label_dir is not None:
            self.dir = label_dir
        else:
            head, tail = os.path.split(self.rel_image.dir)
            return os.path.join(head, tail.replace('.tif', '_label.tif'))


    def check_exist(self):
        """
        Check if label file exist.
        Returns:
            True if successful, False otherwise.
        """
        if os.path.isfile(self.dir):
            return True

        return False


    def get_data(self):
        """
        Return label data.
        """
        if self.data:
            return self.data
        else:
            self.create_label_data()


    def redraw(self):
        """
        check if redraw is necessary
        """
        if (self.exist and self.update):
            return True
        elif (self.exist and not self.update):
            return False
        else:
            return True


    def load_data(self):
        """
        Load label data
        """
        self.data = cv2.imread(self.dir)


    def create_label_data(self):
        """
        Create new label image or load exists
        """
        if self.redraw():
            data = self.rel_image.data.copy()
            rois = self.roi.get_rois()
            for _type in ('background', 'cytoplasm', 'nuclei', 'border'):
                if _type == 'border': # FIXME
                    xy_list = rois['nuclei']
                else:
                    xy_list = rois[_type]
                ds = self.ds[_type]
                for xy in xy_list:
                    cv2.drawContours(data, [xy], ds[0], ds[1], ds[2])
            self.data = data
        else:
            self.load_data()


    def save_label_image(self, save_dir=None):
        """
        Save rel_image with label colors
        """
        if self.data is None:
            self.create_label_data()
        if save_dir is None:
            save_dir = self.dir
        cv2.imwrite(save_dir, self.data)


class Mask:
    """
    Mask class, saved as numpy npy file
    """
    def __init__(self, rel_image, rel_label=None, update=True, 
                 mask_dir=None):
        self.rel_image = rel_image
        self.rel_label = rel_label
        self.dir = self.create_dir(mask_dir)
        self.exist = self.check_exist()
        self.update = update
        self.data = None
        self.ds = {'background': [1, (255, 0, 0)], # FIXME
                   'cytoplasm': [2, (0, 255, 0)],
                   'nuclei': [3, (0, 0, 255)],
                   'border': [4, (0, 0, 0)]}


    def create_dir(self, mask_dir):
        """
        Create label dir
        """
        if mask_dir is not None:
            self.dir = mask_dir
        else:
            head, tail = os.path.split(self.rel_image.dir)
            return os.path.join(head, tail.replace('.tif', '_mask.npy'))


    def check_exist(self):
        """
        Check if label file exist.
        Returns:
            True if successful, False otherwise.
        """
        if os.path.isfile(self.dir):
            return True

        return False


    def get_data(self):
        """
        Return label data.
        """
        if self.data is None:
            self.create_mask_data()
        else:
            return self.data

    def redraw(self):
        """
        check if redraw is necessary
        """
        if (self.exist and self.update):
            return True
        elif (self.exist and not self.update):
            return False
        else:
            return True


    def load_rel_label(self, label):
        """
        Load label data
        If self.label == None will try find label in default place,
        if self.label is path will create new label object with image from path
        if self.label is object then no problem...
        """
        # FIXME
        if isinstance(label, Label):
            self.rel_label = label
        elif label is None:
            try:
                self.rel_label = Label(self.rel_image)
            except:
                print('Problem 1 with create related label!')
        elif os.path.isfile(label):
            self.rel_label = Label(self.rel_image, label_dir=label)
        else:
            print('Problem with 2 create related label!')


    def create_mask_data(self):
        """
        Create new label image or load exists
        """
        if self.redraw():
            data = self.rel_label.data.copy()
            mask = np.zeros_like(data[:,:,0], dtype='int')
            for _type in ('background', 'cytoplasm', 'nuclei', 'border'):
                ds = self.ds[_type]
                color_mask = np.all(data == ds[1], axis=-1)
                mask[color_mask] = ds[0]
            self.data = mask
        else:
            self.load_data()


    def load_data(self):
        self.data = np.load(self.dir)


    def save_mask(self, save_dir=None):
        """
        Save mask 
        """
        if self.data is None:
            self.create_mask_data()
        if save_dir is None:
            save_dir = self.dir
        
        np.save(save_dir, self.data)
