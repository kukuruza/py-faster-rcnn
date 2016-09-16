"""Evaluate net."""

import _init_paths
from datasets.factory import get_imdb
import argparse
import sqlite3
import os, sys, os.path as op


def main(args_list):

    parser = argparse.ArgumentParser(description='Evaluate Fast R-CNN network')
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test on',
                        default='vehicle', type=str)
    parser.add_argument('--gt_db_path', required=True,
                        help='full path to the ground truth .db file')
    parser.add_argument('--det_db_path', required=True,
                        help='full path to the detected .db file')
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    args = parser.parse_args()

    imdb = get_imdb(args.imdb_name, args.gt_db_path)
    imdb.competition_mode(args.comp_mode)

    conn_det = sqlite3.connect (args.det_db_path)
    imdb.evaluate_detections(conn_det.cursor())
    conn_det.close()


if __name__ == '__main__':
    '''Wrap train_net in order to call script from python as well as console.'''
    args_list = sys.argv[1:]
    main(args_list)
