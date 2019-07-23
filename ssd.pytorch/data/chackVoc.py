import argparse
import sys
import cv2
import os
import os.path as osp
import numpy as np
from lxml import etree

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--root', default='./VOCdevkit/',help='Dataset root directory path')
args = parser.parse_args()

CLASSES = ['person']

def vocChecker(annopath, width, height, keep_difficult = False):
    target = etree.parse(annopath).getroot()
    res    = []

    for obj in target.iter('object'):

        difficult = int(obj.find('difficult').text) == 1

        #if not keep_difficult and difficult:
        #    continue

        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        pts    = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []

        for i, pt in enumerate(pts):

            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            cur_pt = float(cur_pt) / width if i % 2 == 0 else float(cur_pt) / height
            bndbox.append(cur_pt)

        #print(name)
        label_idx = dict(zip(CLASSES, range(len(CLASSES))))[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
    #print(res)
    try :
        print(np.array(res)[:,4])
        print(np.array(res)[:,:4])
    except IndexError:
        print("\nINDEX ERROR HERE !\n")
        exit(0)
    return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

if __name__ == '__main__' :

    ids = list()
    rootpath = osp.join(args.root, 'VOC2007')
    for line in open(osp.join(rootpath, 'ImageSets', 'Main', 'trainval.txt'),encoding="utf-8"):
        ids.append(line.strip('\n'))
    for img_id in ids:
        annopath = osp.join(rootpath, 'Annotations' ,img_id[:-4]) + '.xml'
        img = cv2.imread(osp.join(rootpath, 'JPEGImages' ,img_id))
        height, width, channels = img.shape
        print("path : {}".format(annopath))
        res = vocChecker(annopath, height, width)