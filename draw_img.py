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

if not os.path.exists('draw_{}'.format(sys.argv[1])):
    os.mkdir('draw_{}'.format(sys.argv[1]))
for classname in range(3):
    classname = str(classname)
    size = 1280 * 720
    print("class:  {}    ______________________".format(classname))
    res_path = 'train.list'
    with open(res_path, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #classname = '0'
    recs = {}
    for i, imagename in enumerate(imagenames):

        recs[imagename.split('/')[-1]] = parse_gtx(imagename)
        # if i % 100 == 0:
        #     print 'Reading annotation for {:d}/{:d}'.format(
        #         i + 1, len(imagenames))

    class_recs = [{} for i in range(4)]
    npos = [0,0,0,0]
    for imagename in imagenames:
        imagename = imagename.split('/')[-1]
        R = [[] for i in range(4)]
        bbox = [[] for i in range(4)]
        det = [[] for i in range(4)]
        img_name = imagename[:-3]+'jpg'
        for obj in recs[imagename]:
            if obj['name'] == classname:
                if obj['name'] == classname and (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1]) >= size * 0.1:
                    R[0].append(obj)
                elif obj['name'] == classname and size * 0.1 > (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1]) >= size * 0.05:
                    R[1].append(obj)
                elif obj['name'] == classname and size * 0.05 > (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1]) >= size * 0.02:
                    R[2].append(obj)
                elif obj['name'] == classname and (obj['bbox'][2] - obj['bbox'][0]) * (obj['bbox'][3]-obj['bbox'][1]) < size * 0.02:
                    R[3].append(obj)
        for i in range(len(R)):
            bbox[i] = np.array([x['bbox'] for x in R[i]])
            det[i]  = [False] * len(R[i] )
            npos[i]  += len(R[i] )
            class_recs[i][imagename] = {'bbox': bbox[i], 'det': det[i]}


    val_path = 'val_output'
    BB = [ [] for i in range(4)] # all bbox of one class
    confidence = [[] for i in range(4)]
    image_ids = [[] for i in range(4)]
    for val_file in os.listdir(val_path):
        file_path = os.path.join(val_path,val_file)

        prototxt = demo_out_pb2.DetObjs()
        with open(file_path, 'r') as f:
            text_format.Merge(f.read(), prototxt)
        for obj in prototxt.objs:
            if obj.label == int(classname) and obj.bbox.w * obj.bbox.h >= size * 0.1:
                BB[0].append(obj.bbox)
                confidence[0].append(obj.score)
                image_ids[0].append(val_file)
            elif obj.label == int(classname) and size * 0.1 > obj.bbox.w * obj.bbox.h >= size * 0.05:
                BB[1].append(obj.bbox)
                confidence[1].append(obj.score)
                image_ids[1].append(val_file)
            elif obj.label == int(classname) and size * 0.05 > obj.bbox.w * obj.bbox.h >= size * 0.02:
                BB[2].append(obj.bbox)
                confidence[2].append(obj.score)
                image_ids[2].append(val_file)
            elif obj.label == int(classname) and obj.bbox.w * obj.bbox.h < size * 0.02:
                BB[3].append(obj.bbox)
                confidence[3].append(obj.score)
                image_ids[3].append(val_file)
    for i in range(4):
        if npos[i] == 0:
            print('No GroundTruth, Total Predict = {}'.format(len(BB)))
        else:
            #_confidence = np.array(confidence[i])
            #sorted_ind = np.argsort(-_confidence)
            #BB = BB[sorted_ind, ]
            #image_ids = [image_ids[x] for x in sorted_ind]
            nd = len(image_ids[i])
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                img_name = image_ids[i][d][:-9]
                im = Image.open(os.path.join('draw_GT', img_name))
                draw = ImageDraw.Draw(im)
                R = class_recs[i][image_ids[i][d][:-12]+'gtx']
                bc = BB[i][d]
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    bb = []
                    bb.append(bc.x)
                    bb.append(bc.y)
                    bb.append(bc.w+bc.x)
                    bb.append(bc.h+bc.y)
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    # union
                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = np.max(overlaps)  # IOU最大的ovmax
                    jmax = np.argmax(overlaps)  # IOU最大的ovmax的索引

                if ovmax > 0.5:
                    if not R['det'][jmax]:  # 对于一个BBGT，只取符合IOU条件的置信度最高的预测的bb
                        tp[d] = 1.
                        R['det'][jmax] = 1
                        draw.rectangle(bb, outline='green')

                    else:
                        fp[d] = 1.
                        draw.rectangle(bb, outline='yello')
                else:
                    fp[d] = 1.
                    draw.rectangle(bb, outline='red')
                draw.text([bb[0],bb[1]], classname + ":" + str(confidence[i][d]))
                del draw

                # write to stdout
                im.save(os.path.join('draw_{}'.format(sys.argv[1]), img_name), "PNG")
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)

            rec = tp / float(npos[i])
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, use_07_metric=False)
            table = PrettyTable(['Variable', 'Result'])
            table.add_row(['True Predict', int(tp[-1])])
            table.add_row(['False Predict', int(fp[-1])])
            table.add_row(['Total Predict', len(fp)])
            table.add_row(['Total GroundTruth', npos[i]])
            table.add_row(['Recall', rec[-1]])
            table.add_row(['Precision', prec[-1]])
            table.add_row(['AP', ap])
            print(table)
            with open('table.txt' , 'a+') as f:
                f.write(str(table))
#     return ap
# ap0 = voc_eval('0')
# ap1 = voc_eval('1')
# # ap2 = voc_eval('2')
# if ap2 == 0:
#     print('mAP: {}'.format((ap0+ap1)/2))
# else:
#     print('mAP: {}'.format((ap0+ap1+ap2)/3))