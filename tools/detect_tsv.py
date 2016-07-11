#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys
import cv2
import argparse
import base64

CLASSES = ('no-logo', # always index 0
         'adidas', 'aldi', 'apple', 'becks', 'bmw', 'carlsberg', 
         'chimay', 'cocacola', 'corona', 'dhl', 'erdinger', 
         'esso', 'fedex', 'ferrari', 'ford', "fosters", 
         'google', 'guiness', 'heineken', 'hp', 'milka', 
         'nvidia', 'paulaner', 'pepsi', 'rittersport', 'shell', 
         'singha', 'starbucks', 'stellaartois', 'texaco', 
         'tsingtao', 'ups')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

saved_id = 0

def vis_detections(im, url, truecls, detclass, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    global saved_id
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    im1 = im.copy()
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # write the result into the file
        bbox_str = '\t'.join([str(x) for x in bbox])
        f_out.write('%s\t%s\t%s\t%s\t%f\n' % (url, truecls, detclass, bbox_str, score))
        continue

        cv2.rectangle(im1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 4)
        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )
        cv2.putText(im1, '{:s} {:.3f}'.format(detclass, score), 
                    (int(bbox[0]), int(bbox[1]-5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255,255,255), thickness=3)
        cv2.putText(im1, '{:s} {:.3f}'.format(detclass, score), 
                    (int(bbox[0]), int(bbox[1]-5)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,255), thickness=2)
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(detclass, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(detclass, detclass,
    #                                               thresh),
    #               fontsize=14)
    # plt.axis('off')
    # plt.tight_layout()
    # plt.draw()
    # plt.savefig('D:\Users\evgeny\Data\MineBing\detections_100\%06d.jpg' % saved_id)
    print 'im shape: %s' % str(im1.shape)
    cv2.imwrite('D:\Users\evgeny\Data\MineBing\detections_100\%06d.jpg' % saved_id, im1)
    saved_id += 1


    #data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #cv2.imshow('detections', data)
    #cv2.waitKey()

def demo(net, im, f_out, url, truecls):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, detcls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, url, truecls, detcls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--in_tsv_path', help='TSV file with 64bit encoded image data',
                        required=True)
    parser.add_argument('--out_tsv_path', help='TSV file to write detection results to',
                        required=True)
    parser.add_argument('--col', help='image string column in TSV file',
                        required=True, type=int)
    parser.add_argument('--num', help='number of images in TSV file to process',
                        default=1000000, type=int)
    parser.add_argument('--prototxt', help='Path with then net model',
                        required=True)
    parser.add_argument('--caffemodel', help='File path with learned weights',
                        required=True)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if not os.path.isfile(args.caffemodel):
        raise IOError(('{:s} not found').format(args.caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(args.caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    with open(args.in_tsv_path) as f_in:
        with open(args.out_tsv_path, 'w') as f_out:
            f_out.write('url\ttrue_cls\tdet_class\tx\ty\twidth\theight\tscore\n')

            for i in range(args.num):

                # read line
                line = f_in.readline().strip()
                if line == '': break  # EOF or blank line

                lineparts = line.split('\t')
                url = lineparts[2]
                logopart = lineparts[0]
                for word in logopart.split(' '):
                    if word in CLASSES: truecls = word
                print 'processing image %d of class %s' % (i, truecls)

                # find image string
                assert args.col < len(lineparts), \
                    '%d columns >= imagestring col %d' % (len(lineparts), args.col)
                imagestring = lineparts[args.col]

                # decode image string
                try:
                    jpgbytestring = base64.b64decode(imagestring)

                    nparr = np.fromstring(jpgbytestring, np.uint8)
                    img = cv2.imdecode(nparr, cv2.CV_LOAD_IMAGE_COLOR)

                    # cv2.imshow('test', img)
                    # cv2.waitKey()

                except Exception:
                    print 'failed with line %d. Continue' % i
                    continue

                demo(net, img, f_out, url, truecls)
                #plt.show()

