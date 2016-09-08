# Usage:
# python -u ./experiments/scripts/faster_rcnn_end2end.py ^
#   --GPU gpu --NET net --DATASET dataset [cfg args to {train,test}_net.py]
# where DATASET is either pascal_voc or coco.
#
# Example:
# python -u experiments/scripts/faster_rcnn_end2end.py ^
#   --GPU 0 --NET VGG_CNN_M_1024 --DATASET pascal_voc ^
#   EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"
#
# Notes:
#   1) the line-continuation symbol is ^ for cmd, use ` for powershell.
#   2) "-u" flag stands for unbuffered std output

import os, os.path as op
from os.path import dirname as parent
import argparse
from datetime import datetime
import sys
import time

FRCN_ROOT = parent(parent(parent(op.realpath(__file__))))

def at_fcnn(x):
  '''Convenience function to  specify relative paths in code
  Args: x -- path relative to FRCN_ROOT'''
  # op.realpath will take care of the mixed Windows and Unix delimeters '/' and '\'
  return op.realpath(op.join(FRCN_ROOT, x))

# add 'tools' dir to the path
sys.path.insert(0, at_fcnn('tools'))
import train_net, test_net


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--SOLVER_NAME', default='solver.prototxt', help='name of solver file')
  parser.add_argument('--GPU', default='0', type=str, help='GPU id, or -1 for CPU')
  parser.add_argument('--NET', required=True, help='CNN archiutecture, e.g. "VGG16"')
  parser.add_argument('--DATASET', required=True, '"pascal_voc", "coco", etc.')
  parser.add_argument('--DATA_PATH', required=True)
  parser.add_argument('--ITERS',
  	                  help='number of iter., default for pascal_voc = 70K, for coco = 490K')
  parser.add_argument('--LOG_TO_SCREEN', action='store_true', 
                      help='Send stdout to screen instead of the log file for debugging')
  parser.add_argument('EXTRA_ARGS', nargs='*',
                      help='optional cfg for {train,test}_net.py (without "--")')

  args = parser.parse_args()

  EXTRA_ARGS_SLUG = '_'.join(args.EXTRA_ARGS)

  if args.DATASET == 'pascal_voc':
    TRAIN_IMDB = "voc_2007_trainval"
    TEST_IMDB = "voc_2007_test"
    PT_DIR = "pascal_voc"
    ITERS = args.ITERS if args.ITERS is not None else 70000
  elif args.DATASET == 'coco':
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB = "coco_2014_train"
    TEST_IMDB = "coco_2014_minival"
    PT_DIR = "coco"
    ITERS = args.ITERS if args.ITERS is not None else 490000
  elif args.DATASET == 'flickrlogo32':
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMAGESET = "val"
    PT_DIR = "flickrlogo32"
    ITERS = args.ITERS if args.ITERS is not None else 10000
  elif args.DATASET == 'vehicle':
    IMDB = 'vehicle'
    TRAIN_IMAGESET = 'train'
    PT_DIR = 'vehicle'
    ITERS = args.ITERS
  else:
    print 'Provide a dataset, "pascal_voc", "flickrlogo32" or "coco"'
    sys.exit()

  splits = op.splitext(args.SOLVER_NAME)[0].split('-')
  suffix = splits[-1] if len(splits) > 1 else ''
  print 'suffix: %s' % suffix

  # run training
  start = time.time()
  train_net.main([
    '--gpu', args.GPU, 
    '--solver', at_fcnn('models/%s/%s/faster_rcnn_end2end/%s' % 
                        (PT_DIR, args.NET, args.SOLVER_NAME)),
    '--weights', at_fcnn('data/imagenet_models/%s.v2.caffemodel' % args.NET),
    '--image_set', TRAIN_IMAGESET,
    '--imdb', args.DATASET,
    '--data_path', args.DATA_PATH,
    '--iters', str(ITERS),
    '--suffix', suffix,
    '--cfg', at_fcnn('experiments/cfgs/faster_rcnn_end2end.yml')] +
    ([] if not args.EXTRA_ARGS else ['--set'] + args.EXTRA_ARGS))
  print 'tools/train_net.py finished in %s seconds' % (time.time() - start)

