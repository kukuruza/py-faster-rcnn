import os, os.path as op
import argparse

parser = argparse.ArgumentParser(description='Faster R-CNN demo')
parser.add_argument('--in_det_path', required=True,
                    help='TSV file with detections')
parser.add_argument('--out_filt_path', required=True,
                    help='TSV file to write detection results to')

args = parser.parse_args()

with open(args.in_det_path) as f_in:
    lines = f_in.read().split('\n')
    print len(lines)

with open(args.out_filt_path, 'w') as f_out:
    f_out.write('url\ttrue_cls\tdet_cls\tx\ty\twidth\theight\tscore\n')
    for line in lines[1:]:
        if not line: continue
        words = line.split('\t')
        assert len(words) > 1, line
        true_cls = words[1]
        det_cls = words[2]
        if true_cls == det_cls:
            f_out.write(line + '\n')
