import os, os.path as op
import numpy as np


def voc_ap(rec, prec):
  """ ap = voc_ap(rec, prec)
  Compute VOC AP given precision and recall.
  """

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


def eval_class(c_gt,
               c_det,
               classname=None,
               ovthresh=0.5):
  """
  Top level function that does the PASCAL VOC evaluation.

  c_gt  -- cursor to the sqlite3 db with ground truth
  c_out -- cursor to the sqlite3 db with detections
  classname: Category name (duh). If None, then any category
  [ovthresh]: Overlap threshold (default = 0.5)
  """

  if classname is None:
    c_det.execute('SELECT imagefile,x1,y1,width,height,score FROM cars '
                  'ORDER BY score DESC')
  else:
    c_det.execute('SELECT imagefile,x1,y1,width,height,score FROM cars '
                  'ORDER BY score DESC '
                  'WHERE name=?', (classname,))
  car_entries = c_det.fetchall()
  print 'Total %d car_entries of class %s' % (len(car_entries), classname)

  # go down dets and mark TPs and FPs
  tp = np.zeros(len(car_entries), type=float)
  fp = np.zeros(len(car_entries), type=float)

  # 'already_detected' used to penalize multiple detections of same GT box
  if classname is None:
    c_gt.execute('SELECT COUNT(*) FROM cars WHERE name=?', (classname,))
  else:
    c_gt.execute('SELECT COUNT(*) FROM cars')
  npos = c_gt.getchone()[0]
  already_detected = np.zeros(npos, type=float)

  # go through each detection
  for idet,(imagefile,x1,y1,width,height,score) in enumerate(car_entries):

    bbox_det = np.array([x1,y1,width,height], type=float)

    # get all GT boxes from the same imagefile [of the same class]
    if classname is None:
      c_gt.execute('SELECT id,x1,y1,width,height FROM cars '
                     'WHERE imagefile=?', (imagefile,))
    else:
      c_gt.execute('SELECT id,x1,y1,width,height FROM cars '
                     'WHERE imagefile=? AND name=?', (imagefile,classname))
    entries_gt = c_gt.fetchall()
    bboxes_gt = np.array([list(bbox) for bbox in entries_gt[1:]], type=float)

    # separately manage no GT boxes
    if bboxes_gt.size == 0:
      fp[idet] = 1.
      continue

    # intersection
    ixmin = np.maximum(bboxes_gt[0], bbox_det[0])
    iymin = np.maximum(bboxes_gt[1], bbox_det[1])
    ixmax = np.minimum(bboxes_gt[:,0]+bboxes_gt[:,2], bbox_det[0]+bbox_det[2])
    iymax = np.minimum(bboxes_gt[:,1]+bboxes_gt[:,3], bbox_det[1]+bbox_det[3])
    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)
    inters = iw * ih

    # union
    uni = bbox_det[2] * bbox_det[3] + bboxes_gt[:,2] * bboxes_gt[:,3] - inters

    # overlaps
    overlaps = inters / uni
    max_overlap = np.max(overlaps)
    imax_overlap = np.argmax(max_overlap)

    # if 1) large enough overlap and 2) this GT box was not detected before
    if max_overlap > ovthresh and not already_detected[imax_overlap]:
      tp[idet] = 1.
      already_detected[imax_overlap] = 1.
    else:
      fp[idet] = 1.

  # compute precision-recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap
