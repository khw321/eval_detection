# -*- coding: UTF-8 -*-
import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
from google.protobuf import text_format
import demo_out_pb2
from prettytable import PrettyTable
import sys

from PIL import Image,ImageDraw



def parse_gtx(filename):
    objects = []
    #gtx_path = os.path.join('cbox_data', filename)
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(';')
            obj_struct = {}
            obj_struct['name'] = line[0]
            obj_struct['bbox'] = [int(line[1]),int(line[2]),int(line[3]),int(line[4]),]
            objects.append(obj_struct)

    return objects

if not os.path.exists('draw_{}'.format(sys.argv[1])):
    os.mkdir('draw_{}'.format(sys.argv[1]))
size = 1280 * 720
res_path = 'train.list'
with open(res_path, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]
recs = {}
for i, imagename in enumerate(imagenames):

    recs[imagename.split('/')[-1]] = parse_gtx(imagename)
    # if i % 100 == 0:
    #     print 'Reading annotation for {:d}/{:d}'.format(
    #         i + 1, len(imagenames))


for imagename in imagenames:
    imagename = imagename.split('/')[-1]
    img_name = imagename[:-3]+'jpg'
    im = Image.open(os.path.join('raw_images', img_name))
    draw = ImageDraw.Draw(im)
    for obj in recs[imagename]:
        draw.rectangle([obj['bbox'][0],obj['bbox'][1],obj['bbox'][2],obj['bbox'][3]], outline='blue')
        draw.text([obj['bbox'][0],obj['bbox'][1]], obj['name'])
    del draw
    im.save(os.path.join('draw_{}'.format(sys.argv[1]), img_name), "jpeg")