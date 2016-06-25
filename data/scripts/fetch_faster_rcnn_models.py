#!/bin/python
from fetch import fetch

FILE = 'faster_rcnn_models.tgz'
URL = 'http://people.eecs.berkeley.edu/~rbg/faster-rcnn-data/%s' % FILE
CHECKSUM = 'ac116844f66aefe29587214272054668'

fetch(FILE, URL, CHECKSUM)
