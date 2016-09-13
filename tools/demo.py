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
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

CLASSES = ('__background__', 'vehicle')

def vis_detections(im, class_name, dets, thresh):
    """Draw detected bounding boxes."""

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for det in dets:
        bbox = det[:4]
        score = det[-1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, im_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)

    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        # filter out nms
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        print 'after NMS left %d' % dets.shape[0]
        # filter out by confidence
        keep = inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        dets = dets[keep, :]
        print 'after confidence left %d' % dets.shape[0]
        vis_detections(im, cls, dets, CONF_THRESH)
    print 'has %d boxes' % dets.shape[0]
    print dets[:,:4]

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--prototxt', required=True,
      help='Path to architecture (prototxt), '
           'e.g. models/pascal_voc/VGG16/faster_rcnn_end2end/train.prototxt"')
    parser.add_argument('--caffemodel', required=True,
      help='Path to trained net (caffemodel), e.g. '
           '"output/faster_rcnn_end2end/vehicle_train/iter_25000.caffemodel" '
           'or "data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel"')
    parser.add_argument('--im_names', action='append',
      help='Paths to images to test, default is '
           '[000456.jpg, 000542.jpg, 001150.jpg, 001763.jpg, 004545.jpg]')
    parser.add_argument('--set', dest='set_cfgs', nargs=argparse.REMAINDER,
      help='set config keys', default=None)
    args = parser.parse_args()

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if not os.path.isfile(args.caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.py?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(args.caffemodel)

    # Warmup on a dummy image
    if not args.cpu_mode:
        im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
        for i in xrange(2):
            _, _= im_detect(net, im)

    im_names = args.im_names
    if not im_names:
        im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                    '001763.jpg', '004545.jpg']
        im_names = [os.path.join(cfg.DATA_DIR, 'demo', x) for x in im_names]
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
