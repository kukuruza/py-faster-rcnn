"""Evaluate net."""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import sqlite3
import time, os, sys, os.path as op

def parse_args(args_list):
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test on',
                        default='vehicle', type=str)
    parser.add_argument('--gt_db_path', required=True,
                        help='full path to the ground truth .db file')
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--det_db_path', default=':memory:',
                        help='full path to the detected .db file')

    if len(args_list) == 0:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args_list)
    return args

def main(args_list):
    args = parse_args(args_list)

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    assert op.exists(args.caffemodel), '%s does not exist' % args.caffemodel

    if cfg.GPU_ID < 0:
        print 'Setting CPU mode'
        caffe.set_mode_cpu()
        cfg.USE_GPU_NMS = False
    else:
        print 'Setting GPU device %d for training' % cfg.GPU_ID
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    imdb = get_imdb(args.imdb_name, args.gt_db_path)
    #imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(args.comp_mode)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    conn_det = sqlite3.connect (args.det_db_path)
    imdb.evaluate_detections(conn_det.cursor())
    conn_det.close()


if __name__ == '__main__':
    '''Wrap train_net in order to call script from python as well as console.'''
    args_list = sys.argv[1:]
    main(args_list)