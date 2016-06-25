#!/bin/python
from fetch import fetch

FILE = 'selective_search_data.tgz'
URL = 'http://people.eecs.berkeley.edu/~rbg/faster-rcnn-data/%s' % FILE
CHECKSUM = '7078c1db87a7851b31966b96774cd9b9'

fetch(FILE, URL, CHECKSUM)