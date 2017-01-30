import numpy as np
import os
from read_roi import read_roi_zip

class Image:
    """
    Image class
    """
    def __init__(self, im_dir, rois_dir=None, label_dir=None, mask_dir=None):
        self.dir = im_dir
        self.data = self.load_data()
        self.tumor_type = None
        self.rois = {}
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
        import cv2
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
        if self.rois:
            return self.rois
        else:
            roi = Roi(self, self.rois_dir)
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
            if (self.nuclei, self.cytoplasm, self.background) is not None:
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

            for r in roi_array: # FIXME
                xy = []
                for x, y in zip(r['x'], r['y']):
                    xy.append([x, y])
                xy_list.append(np.array(xy))
            setattr(self, _type, xy_list)


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
                   'nuclei': (-1, (0, 0, 255), 2)}

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
        print(self.dir)
        if os.path.isfile(self.dir):
            return True

        return False


    def get_data(self):
        """
        Return label data.
        """
        import cv2
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
        import cv2
        if self.redraw():
            print(self.rel_image.data)
            data = self.rel_image.data.copy()
            for _type in ('background', 'cytoplasm', 'nuclei'):
                xy_list = self.roi.rois[_type]
                for xy in xy_list:
                    cv2.drawContours(data, [xy], self.ds[0], self.ds[1],
                                     self.ds[2])
            cv2.imwrite(self.dir, data)
            self.data = data
        else:
            self.load_data()
