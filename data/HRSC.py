import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform


if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET



import os

import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET




class HRSCDataset(torch.utils.data.Dataset):


    def __init__(self, data_dir, use_difficult=False, transform=None):
        self.root = data_dir

        self.keep_difficult = use_difficult
        self.transform  = transform

        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "AllImages", "%s.bmp")
        self._imgset = os.listdir(os.path.join(self.root, "AllImages"))
        self.ids = [x.split('.')[0] for x in self._imgset]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}


    def load_image(self, image_index):
        img_id = self.ids[image_index]
        #img = Image.open(self._imgpath % img_id)
        img = skimage.io.imread(self._imgpath % img_id)
        # if len(img.shape) == 2:
        #     img = skimage.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)

    def load_annotations(self, image_index):
        img_id = self.ids[image_index]
        anno = self._preprocess_annotation(self._annopath % img_id)['ann']
        annotations = np.array(anno["boxes"]).astype(np.float32)

        return annotations


    def _preprocess_annotation(self, xml_path):

        tree = ET.parse(xml_path)
        root = tree.getroot()
        # size = root.find('Size')
        w = int(root.find('Img_SizeWidth').text)
        h = int(root.find('Img_SizeHeight').text)
        bboxes = []
        labels = []
        objects = root.find('HRSC_Objects')
        for obj in objects.findall('HRSC_Object'):
            difficult = int(obj.find('difficult').text)
            # bnd_box = obj.find('bndbox')
            bbox = [
                float(obj.find('box_xmin').text),
                float(obj.find('box_ymin').text),
                float(obj.find('box_xmax').text),
                float(obj.find('box_ymax').text),
            ]

            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        annotation = {
            'filename': root.find('Img_FileName').text+'.bmp',
            'width': w,
            'height': h,
            'ann': {
                'boxes': bboxes.astype(np.float32),
                # 'bboxes_ignore': bboxes_ignore.astype(np.float32),
                # 'labels_ignore': labels_ignore.astype(np.int64)
            }
        }
        return annotation


    def image_aspect_ratio(self, image_index):
        img_id = self.ids[image_index]
        anno = ET.parse(self._annopath % img_id).getroot()

        im_info = tuple(map(int, (anno.find("Img_SizeWidth").text, anno.find("Img_SizeHeight").text)))
        w = im_info[0]
        h = im_info[1]
        #image = Image.open(self.image_names[image_index])
        return float(w) / float(h)
