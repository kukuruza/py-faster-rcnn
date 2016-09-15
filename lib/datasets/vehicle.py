import os, os.path as op
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import subprocess
import uuid
from db_eval import eval_class
from fast_rcnn.config import cfg
import sqlite3
import sys

class vehicle(imdb):
  ''' Binary classifier vehicles / nonvehicles '''

  def __init__(self, db_path):
    imdb.__init__(self, op.basename(db_path))

    assert op.exists(db_path), 'db_path does not exist: %s' % db_path
    self.conn = sqlite3.connect (db_path)
    self.c    = self.conn.cursor()

    self._classes = ('__background__', 'vehicle')
    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
    # Default to roidb handler
    self._roidb_handler = self.selective_search_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    self.config = {'cleanup'     : True,
                   'use_salt'    : True,
                   'use_diff'    : False,
                   'matlab_eval' : False,
                   'rpn_file'    : None,
                   'min_size'    : 2}


  def num_images(self):
    self.c.execute('SELECT COUNT(imagefile) FROM images')
    return self.c.fetchone()[0]


  def _get_widths(self):
    self.c.execute('SELECT width FROM images')
    return [width for (width,) in self.c.fetchall()]


  def get_imagefile_at(self, i):
    self.c.execute('SELECT imagefile FROM images')
    return self.c.fetchall()[i][0]


  def get_imagefiles(self):
    self.c.execute('SELECT imagefile FROM images')
    for imagefile, in self.c.fetchall()[i]:
      yield imagefile


  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    """
    gt_roidb = []

    self.c.execute('SELECT imagefile FROM images')
    for (imagefile,) in self.c.fetchall():

      self.c.execute('SELECT x1,y1,width,height FROM cars '
                     'WHERE imagefile=?', (imagefile,))
      entries = self.c.fetchall()
      num_objs = len(entries)

      boxes = np.zeros((num_objs, 4), dtype=np.uint16)
      gt_classes = np.zeros((num_objs), dtype=np.int32)
      overlaps = np.zeros((num_objs, 2), dtype=np.float32) # '__background__' & 'vehicle'
      # "Seg" area for pascal is just the box area
      seg_areas = np.zeros((num_objs), dtype=np.float32)
      cls_inds = []

      # Load object bounding boxes into a data frame.
      for ix, (x1,y1,width,height) in enumerate(entries):
          x2 = x1 + width
          y2 = y1 + height
          #cls = self._class_to_ind[obj.find('name').text.lower().strip()]
          cls_inds.append(ix)  # need only our class
          boxes[ix, :] = [x1, y1, x2, y2]
          gt_classes[ix] = 1  # 1 is the 'vehicle' index
          overlaps[ix, 1] = 1.0  # 1 is the 'vehicle' index
          seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

      # need only one class
      boxes = boxes[cls_inds, :]
      gt_classes = gt_classes[cls_inds]
      overlaps = overlaps[cls_inds, :]
      seg_areas = seg_areas[cls_inds]

      overlaps = scipy.sparse.csr_matrix(overlaps)

      self.c.execute('SELECT width,height FROM images WHERE imagefile=?', (imagefile,))
      width,height = self.c.fetchone()

      gt_roidb.append(
             {'imagefile': imagefile,
              'boxes' : boxes,
              'width': width,
              'height': height,
              'gt_classes': gt_classes,
              'gt_overlaps' : overlaps,
              'flipped' : False,
              'seg_areas' : seg_areas})
    return gt_roidb


  def selective_search_roidb(self):
    """
    Return the database of selective search regions of interest.
    Ground-truth ROIs are also included.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path,
                              self.name + '_selective_search_roidb.pkl')

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        return roidb

    gt_roidb = self.gt_roidb()
    ss_roidb = self._load_selective_search_roidb(gt_roidb)
    roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
    
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
    print 'wrote ss roidb to {}'.format(cache_file)

    return roidb

  # def rpn_roidb(self):
  #   if self._image_set != 'test':
  #       gt_roidb = self.gt_roidb()
  #       rpn_roidb = self._load_rpn_roidb(gt_roidb)
  #       roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
  #   else:
  #       roidb = self._load_rpn_roidb(None)

  #   return roidb

  # def _load_rpn_roidb(self, gt_roidb):
  #   filename = self.config['rpn_file']
  #   print 'loading {}'.format(filename)
  #   assert os.path.exists(filename), \
  #          'rpn data not found at: {}'.format(filename)
  #   with open(filename, 'rb') as f:
  #       box_list = cPickle.load(f)
  #   return self.create_roidb_from_box_list(box_list, gt_roidb)

#    def _load_selective_search_roidb(self, gt_roidb):
#        # replace flickrlogo1 with flickrlogo32 for selective search
#        idx1 = self.name.find('flickrlogo1')
#        idx2 = self.name.find('_', idx1+len('flickrlogo1_'))
#        name = self.name[:idx1] + 'flickrlogo32' + self.name[idx2:]
#        print name
#        
#        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
#                                                'selective_search_data',
#                                                name + '.mat'))
#        assert os.path.exists(filename), \
#               'Selective search data not found at: {}'.format(filename)
#        raw_data = sio.loadmat(filename)['boxes'].ravel()
#
#        box_list = []
#        for i in xrange(raw_data.shape[0]):
#            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
#            keep = ds_utils.unique_boxes(boxes)
#            boxes = boxes[keep, :]
#            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
#            boxes = boxes[keep, :]
#            box_list.append(boxes)
#
#        return self.create_roidb_from_box_list(box_list, gt_roidb)


  def evaluate_detections(self, all_boxes):
      aps = []
      if not os.path.isdir(output_dir):
          os.mkdir(output_dir)
      for clsid, cls_name in enumerate(self._classes):
          if cls_name == '__background__':
              continue
          rec, prec, ap = eval_class (
              self.c, all_boxes[clsid], cls_name=None, ovthresh=0.5)
          aps += [ap]
          print('AP for {} = {:.4f}'.format(cls_name, ap))
          with open(os.path.join(output_dir, cls_name + '_pr.pkl'), 'w') as f:
              cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
      print('Mean AP = {:.4f}'.format(np.mean(aps)))
      print('~~~~~~~~')
      print('Results:')
      for ap in aps:
          print('{:.3f}'.format(ap))
      print('{:.3f}'.format(np.mean(aps)))
      print('~~~~~~~~')
