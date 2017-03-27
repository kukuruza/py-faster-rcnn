#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test different iterations of trained Fast R-CNN network on an image database."""

import _init_paths
import logging
import argparse
import pprint
import glob
import test_net
import time, os, sys, os.path as op

def parse_args(args_list):
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--net_template', required=True,
                        help='.caffemodel file templates')
    parser.add_argument('--out_db_dir', default=':memory:',
                        help='filepath of output database. Default is in-memory')
    parser.add_argument('--logging_level', default=20, type=int)

    if len(args_list) == 0:
        parser.print_help()
        sys.exit(1)

    return parser.parse_known_args(args_list)


def main(args_list):
    args, args_list_remaining = parse_args(args_list)
    logging.basicConfig(level=args.logging_level)

    net_paths = glob.glob(args.net_template)
    if not net_paths:
        logging.error('Nothing found at %s' % args.net_template)
        sys.exit(1)
    for net_path in net_paths:
        logging.info('Found trained model %s' % net_path)
    
    results = {}

    if args.out_db_dir != ':memory:' and not op.exists(args.out_db_dir):
        os.makedirs(args.out_db_dir)

    for net_path in net_paths:
        net_name = op.splitext(op.basename(net_path))[0]
        logging.info('Testing model %s' % net_name)

        out_db_path = op.join(args.out_db_dir, '%s.db' % net_name) \
                if args.out_db_dir != ':memory:' else ':memory:'

        args_list_net = ['--net', net_path, '--out_db_path', out_db_path] + \
                list(args_list_remaining)

        results[net_name] = test_net.main(args_list_net)

        # printout raw dict
        #pprint.pprint(results)

        # now assume that each name is of format "*_iternum", e.g. "vgg16_iter_10000"
        pretty_results = {}
        for x in results:
            iternum = int(x.split('_')[-1])
            pretty_results[iternum] = results[x]
        for x in sorted(pretty_results.keys()):
            print '%d\t%.04f' % (x, pretty_results[x])

    print 'test_net_iter completed evaluation'
    return pretty_results


if __name__ == '__main__':
    args_list = sys.argv[1:]
    main(args_list)
    
