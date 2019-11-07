import sys
import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import os

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET





def preprocess_annotation(xml_path):
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
        'filename': root.find('Img_FileName').text + '.bmp',
        'width': w,
        'height': h,
        'boxes': bboxes.astype(np.float32),
            # 'bboxes_ignore': bboxes_ignore.astype(np.float32),
            # 'labels_ignore': labels_ignore.astype(np.int64)

    }
    return annotation


xml_path = '/media/zxq/data/data/object_data/HSRC2016/HRSC2016_part01/Test/Annotations'
for xml in os.listdir(xml_path):
    path = os.path.join(xml_path,xml)
    ann = preprocess_annotation(path)
    if np.size(ann['boxes'])==0 :
        print(xml)
