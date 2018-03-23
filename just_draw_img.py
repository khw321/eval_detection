# -*- coding: UTF-8 -*-
import xml.etree.ElementTree as ET
import os
import cPickle
import numpy as np
from google.protobuf import text_format
import demo_out_pb2
from prettytable import PrettyTable
import sys

from PIL import Image
from PIL import ImageDraw



def parse_gtx(filename):
    objects = []
    #gtx_path = os.path.join('cbox_data', filename)
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(';')
            obj_struct = {}
            obj_struct['name'] = line[0]
            obj_struct['bbox'] = [int(line[1]),int(line[2]),int(line[3]),int(line[4]),]
            obj_struct['det'] = 0
            objects.append(obj_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
save_path = 'draw_'+str(sys.argv[1])
if not os.path.exists(save_path):
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'AllTrue_Pre'))
    os.mkdir(os.path.join(save_path, 'False_Pre'))
    os.mkdir(os.path.join(save_path, 'Repeat_Pre'))
    os.mkdir(os.path.join(save_path, 'Miss_GT'))


res_path = 'train.list'
with open(res_path, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip() for x in lines]
val_path = 'val_output'
count = 0
for imagename in imagenames:
    count = count + 1
    if count % 100 == 0:
        print('processd {} images'.format(count))
    flag = [0, 0, 0]
    recs = parse_gtx(imagename) # get all objs in a image GT
    imagename = imagename.split('/')[-1][:-3]+'jpg'
    im = Image.open(os.path.join('draw_GT', imagename))
    img_draw = ImageDraw.Draw(im)
    file_path = os.path.join(val_path, imagename+'.prototxt')
    prototxt = demo_out_pb2.DetObjs()
    with open(file_path, 'r') as f:
        text_format.Merge(f.read(), prototxt)
    for obj in prototxt.objs:   # get all objs in a image Predict
        ovmax = -np.inf
        max_id = -np.inf
        bb=[]
        bb.append(obj.bbox.x)
        bb.append(obj.bbox.y)
        bb.append(obj.bbox.w + obj.bbox.x)
        bb.append(obj.bbox.h + obj.bbox.y)
        for i in range(len(recs)):
            if str(obj.label) == recs[i]['name']:
                ixmin = np.maximum(recs[i]['bbox'][0], bb[0])
                iymin = np.maximum(recs[i]['bbox'][1], bb[1])
                ixmax = np.minimum(recs[i]['bbox'][2], bb[2])
                iymax = np.minimum(recs[i]['bbox'][3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (recs[i]['bbox'][2] - recs[i]['bbox'][0] + 1.) *
                       (recs[i]['bbox'][3] - recs[i]['bbox'][1] + 1.) - inters)

                overlaps = inters / uni
                if overlaps >= ovmax:
                    ovmax = overlaps
                    max_id = i
        if ovmax > 0.5:
            if not recs[max_id]['det']:
                img_draw.rectangle(bb, outline='green')
                img_draw.text([bb[0], bb[3]], str(obj.label) + ':' + str('{:.2f}'.format(obj.score)), fill='green')
                recs[max_id]['det'] = 1
                flag[2] = flag[2] + 1
            else:
                img_draw.rectangle(bb, outline='yellow')
                img_draw.text([bb[0], bb[3]], str(obj.label) + ':' + str('{:.2f}'.format(obj.score)), fill='yellow')
                flag[1] = 1
        else:
            flag[0] = 1
            img_draw.rectangle(bb, outline='red')
            img_draw.text([bb[0], bb[3]], str(obj.label) + ':' + str('{:.2f}'.format(obj.score)), fill='red')
    del img_draw
    # write to stdout
    if flag[0]:
        im.save(os.path.join(save_path, 'False_Pre', imagename), "PNG")
    if flag[1]:
        im.save(os.path.join(save_path, 'Repeat_Pre', imagename), "PNG")
    if flag[0] == 0 and flag[1] == 0:
        im.save(os.path.join(save_path, 'AllTrue_Pre', imagename), "PNG")
    if flag[2] < len(recs):
        im.save(os.path.join(save_path, 'Miss_GT', imagename), "PNG")
        print('miss')


