# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
from utils.blob import im_list_to_blob
import sqlite3

import sys, os
sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperImg import ReaderVideo
from learning.helperDb import createDb, imageField


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = net.blobs['cls_score'].data
    else:
        # use softmax estimated probabilities
        scores = blobs_out['cls_prob']

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, im.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes

def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(net, imdb, thresh=0.05, out_db_path=':memory:'):
  '''Test a Fast R-CNN network on an image database.
  '''
  num_images = imdb.num_images()
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  timer = Timer()

  if not cfg.TEST.HAS_RPN:
    roidb = imdb.roidb

  reader = ReaderVideo()

  if os.path.exists(out_db_path):
    os.remove(out_db_path)
  conn_out = sqlite3.connect(out_db_path)
  createDb(conn_out)
  c_out = conn_out.cursor()

  imdb.c.execute('SELECT * FROM images')
  for imid,image_entry in enumerate(imdb.c.fetchall()):

    imagefile = imageField(image_entry, 'imagefile')
    s = 'images(imagefile,src,width,height,maskfile,time)'
    c_out.execute('INSERT INTO %s VALUES (?,?,?,?,?,?)' % s, image_entry)

    # filter out any ground truth boxes
    if cfg.TEST.HAS_RPN:
      box_proposals = None
    else:
      # The roidb may contain ground-truth rois (for example, if the roidb
      # comes from the training or val split). We only want to evaluate
      # detection on the *non*-ground-truth rois. We select those the rois
      # that have the gt_classes field set to 0, which means there's no
      # ground truth.
      box_proposals = roidb[imid]['boxes'][roidb[imid]['gt_classes'] == 0]

    im = reader.imread(imagefile)
    timer.tic()
    scores, boxes = im_detect(net, im, box_proposals)
    # scores - N x num_classes array
    # boxes  - N x 4 array, each row is [x1, y1, x2, y2]
    timer.toc()

    for clsid in xrange(1, scores.shape[1]):  # clsid = 0 is background

      # apply threshold
      cls_inds = np.where(scores[:, clsid] > thresh)[0]
      cls_scores = scores[cls_inds, clsid]
      cls_boxes = boxes[cls_inds, clsid*4:(clsid+1)*4]
      cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                          .astype(np.float32, copy=False)

      # perform NMS by class
      keep = nms(cls_dets, cfg.TEST.NMS)
      cls_dets = cls_dets[keep, :]

      # save the filtered to the db
      for cls_det in cls_dets:
        score = cls_det.astype(float)[4]
        roi   = cls_det.astype(int)[0:4]
        classname = imdb._classes[clsid]
        s = 'imagefile,name,score,x1,y1,width,height'
        v = (imagefile, classname, score, 
             roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1])
        c_out.execute('INSERT INTO cars(%s) VALUES (?,?,?,?,?,?,?)' % s, v)

    print 'im_detect: {:d}/{:d} {:.3f}s. Found {:d} objects.' \
          .format(imid+1, num_images, timer.average_time, len(cls_dets))

  conn_out.commit()

  print 'Evaluating detections'
  ap = imdb.evaluate_detections(c_det=c_out)

  conn_out.close()
  return ap
