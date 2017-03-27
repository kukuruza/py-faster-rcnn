#!/usr/bin/env python
import sys, os, os.path as op
sys.path.insert(0, op.join(os.getenv('CITY_PATH'), 'src'))
import _init_paths
import caffe
import argparse
import pprint
import logging
import numpy as np
import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import test_net_iters
from datasets.factory import get_imdb
import datasets.imdb
from learning.helperSetup import atcity





def combined_roidb(imdb_name, db_path):
    imdb = get_imdb(imdb_name, db_path)
    logging.info('Loaded dataset `%s` for training' % imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    logging.info('Set proposal method: %s' % cfg.TRAIN.PROPOSAL_METHOD)
    roidb = get_training_roidb(imdb)
    return roidb


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Execute train/test pipeline')
  parser.add_argument('--train_db_dir', required=True,
        help='relative to citycam/faster-rcnn, E.g. 572-Feb23-09h/mar01-synth50+real')
  parser.add_argument('--gpu', default=0, type=int)
  
  parser.add_argument('--iters', type=int, default=30000)
  parser.add_argument('--architecture_name', default='test.prototxt')
  parser.add_argument('--solver_name', default='solver.prototxt')
  parser.add_argument('--cfg', dest='cfg_file',
        help='config cfg file', 
        default='experiments/cfgs/faster_rcnn_end2end.yml')
  parser.add_argument('--train_db_name', default=None,
        help='''name of .db file inside train_db_dir, if different than usual.
                E.g. myfancyname.db''')
  parser.add_argument('--logging_level', default=20, type=int)
  parser.add_argument('--set', dest='set_cfgs',
        help='set config keys', default=None,
        nargs=argparse.REMAINDER)

  args = parser.parse_args()
  
  logging.basicConfig(level=args.logging_level)
  logging.debug(pprint.pformat(args, indent=2))

  architecture_dir = 'models/vehicle1/VGG16/faster_rcnn_end2end'
  solver = op.join(architecture_dir, args.solver_name)
  architecture = op.join(architecture_dir, args.architecture_name)
  pretrained_model = 'data/imagenet_models/VGG16.v2.caffemodel'

  # output dir
  output_dir = op.join('output', args.train_db_dir)
  logging.info('Output will be saved to `%s`' % output_dir)
  if not op.exists(output_dir):
    logging.info('Will create output directory.')
    os.makedirs(output_dir)

  # train_db path
  if args.train_db_name is not None:
    train_db_name = args.train_db_name
  else:
    train_db_name = '%s.db' % op.basename(args.train_db_dir)
  logging.info('train_db_name: %s' % train_db_name)
  train_db_file = atcity(op.join('data/faster-rcnn', args.train_db_dir, train_db_name))
  logging.info('train_db_file: %s' % train_db_file)

  # cfg
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  # random seed
  np.random.seed(cfg.RNG_SEED)
  caffe.set_random_seed(cfg.RNG_SEED)

  # gpu
  assert args.gpu >= 0, 'no cpu mode'
  cfg.GPU_ID = args.gpu
  logging.info('Setting GPU device %d for training' % cfg.GPU_ID)
  caffe.set_mode_gpu()
  caffe.set_device(cfg.GPU_ID)


  # training

#  roidb = combined_roidb('vehicle', train_db_file)
#  logging.info('%d roidb entries' % len(roidb))

#  train_net(solver, roidb, output_dir,
#            pretrained_model=pretrained_model,
#            max_iters=args.iters)


  # testing
  net_template = op.join(output_dir, '*.caffemodel')
  car_constraint = 'width >= 30'
  results_suffix = '-%s' % car_constraint.split(' ')[-1]

  logging.info('Starting testing')
  logging.info('net_template: %s' % net_template)
  logging.info('results_suffix: %s' % results_suffix)

  test_net_iters.main(['--in_db_path', train_db_file,
                       '--net_template', net_template,
                       '--max_images', '100',
                       '--results_suffix', '-w30'] + 
                      sys.argv[1:] + ['TEST.CAR_CONSTRAINT', car_constraint])


