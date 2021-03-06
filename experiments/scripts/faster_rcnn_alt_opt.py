# Usage:
# python -u ./experiments/scripts/faster_rcnn_alt_opt.py ^
#   --GPU gpu --NET net --DATASET dataset [cfg args to {train,test}_net.py]
# where DATASET is only pascal_voc for now.
#
# Example:
# python -u experiments/scripts/faster_rcnn_alt_opt.py ^
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
print at_fcnn('tools')
sys.path.insert(0, at_fcnn('tools'))
import train_faster_rcnn_alt_opt, test_net


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--GPU', default='0', type=str, help='GPU id, or -1 for CPU')
  parser.add_argument('--NET', required=True, help='CNN archiutecture')
  parser.add_argument('--DATASET', required=True, help='either "pascal_voc" or "coco"')
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
  elif args.DATASET == 'coco':
    print "Not implemented: use experiments/scripts/faster_rcnn_end2end.py for coco"
    sys.exit()
  else:
    print 'Provide a dataset, either "pascal_voc" or "coco"'
    sys.exit()

  # redirect output to the LOG file
  if not args.LOG_TO_SCREEN:
    DATE = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    LOG = at_fcnn('experiments/logs/faster_rcnn_alt_opt%s_%s_%s.txt' %
          (args.NET, EXTRA_ARGS_SLUG, DATE))
    print 'Logging output to %s' % LOG
    sys.stdout = open(LOG, 'w')

  # run training
  start = time.time()
  train_faster_rcnn_alt_opt.main([
    '--net_name', args.NET,
    '--gpu', args.GPU, 
    '--weights', at_fcnn('data/imagenet_models/%s.v2.caffemodel' % args.NET),
    '--imdb', TRAIN_IMDB,
    '--cfg', at_fcnn('experiments/cfgs/faster_rcnn_alt_opt.yml')] +
    ([] if not args.EXTRA_ARGS else ['--set'] + args.EXTRA_ARGS))
  print 'tools/train_faster_rcnn_alt_opt.py finished in %s seconds' % (time.time() - start)

  # find the path to the snapshot in the log
  try:
    with open(LOG) as f:
      lines = f.readlines()
      # find a line that has these words
      line = next((line for line in reversed(lines) if 'Wrote snapshot to:' in line), None)
      print 'Taking this line from log to get the snapshot: %s' % line
      # the 4th word in this line is the name of the model
      NET_FINAL = line.split()[3]
      print 'Will test the output model at: %s' % NET_FINAL
  except:
    print 'Cant find "Wrote snapshot to:" line in LOG file.'
    print 'Log file: %s' % LOG
    sys.exit()

  # run testing
  start = time.time()
  test_net.main([
    '--gpu', args.GPU,
    '--def', at_fcnn('models/%s/%s/faster_rcnn_alt_opt/faster_rcnn_test.pt' % (PT_DIR, args.NET)),
    '--net', NET_FINAL,
    '--imdb', TEST_IMDB,
    '--cfg', at_fcnn('experiments/cfgs/faster_rcnn_alt_opt.yml')] + \
    ([] if not args.EXTRA_ARGS is not None else ['--set'] + args.EXTRA_ARGS))
  print 'tools/test_net.py finished in %s seconds' % (time.time() - start)
