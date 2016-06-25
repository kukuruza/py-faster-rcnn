#!/bin/python
from fetch import fetch

FILE = 'imagenet_models.tgz'
URL = 'http://people.eecs.berkeley.edu/~rbg/faster-rcnn-data/%s' % FILE
CHECKSUM = 'ed34ca912d6782edfb673a8c3a0bda6d'

fetch(FILE, URL, CHECKSUM)
