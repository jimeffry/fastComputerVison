# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress

# from mmdet.evaluation import voc_classes

# label_ids = {name: i for i, name in enumerate(voc_classes())}


def parse_xml(args):
    xml_path, img_path, label_ids = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(image_dir, xml_dir, label_names,out_file):
    image_names = os.listdir(image_dir)
    # xml_paths = []
    # img_paths = []
    names_spl = label_names.split(",")
    name_dict = dict()
    for i,tmp_name in enumerate(names_spl):
        name_dict[tmp_name] = i
    annotations = []
    for tmp in image_names:
        tmp_img = osp.join(image_dir,tmp.strip())
        basename_no_extension = osp.splitext(tmp)[0]
        tmp_xml = osp.join(xml_dir,f'{basename_no_extension}.xml')
        # img_paths.append(tmp_img)
        # xml_paths.append(tmp_xml)
        tmp_anno = parse_xml(tmp_xml,tmp_img,name_dict)
        annotations.append(tmp_anno)
    # annotations = track_progress(parse_xml,list(zip(xml_paths, img_paths)))
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    dump(annotations, out_file)
    # return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('--image-dir', type=str,help='pascal voc image path')
    parser.add_argument('--xml-dir', type=str,help='pascal voc xml path')
    parser.add_argument('--data-dir', type=str,help='pascal voc data path')
    parser.add_argument('--out-dir',type=str, help='output path')
    parser.add_argument('--label-names',type=str, type=str,help='pascal voc xml path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # image_dir = args.image_dir
    label_names = args.label_names
    # xml_dir = args.xml_dir
    # out_dir = args.out_dir 
    data_root = args.data_dir
    image_dir = osp.join(data_root,"images")
    xml_dir = osp.join(data_root,"xmls")
    out_dir = osp.join(data_root,"annotations")
    if not osp.exists(image_dir):
        print("please check out the 'images' is existing ... ")
        return
    if not osp.exists((xml_dir)):
        print("please check out the 'xmls' is existing ... ")
        return
    mkdir_or_exist(out_dir)
    dataset_name = image_dir.split("/")[-2]
    outfile = osp.join(out_dir,dataset_name+".json")
    cvt_annotations(image_dir,xml_dir,label_names,outfile)
    print('convert data Done!')


if __name__ == '__main__':
    main()
