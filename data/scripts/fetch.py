#!/bin/python
import urllib
import os, os.path as op
import hashlib
import tarfile

'''
This file provides the function fetch() which downloads and extracts a file from the web.
It is used by fetch_imagenet_models.py, fetch_faster_rcnn_models.py, 
  and fetch_selective_search.py
'''

def verify_checksum(filepath, true_md5):
  calc_md5 = hashlib.md5(open(filepath,'rb').read()).hexdigest()
  if calc_md5 == true_md5:
    print 'The checksum matches'
  else:
    print 'The checksum is WRONG (got %s, true: %s). ' \
          'Please download again.' % (calc_md5, true_md5)
  return calc_md5 == true_md5

def extract_file(filepath, to_directory='.'):
  opener, mode = tarfile.open, 'r:gz'
  cwd = os.getcwd()
  os.chdir(to_directory)

  try:
    file = opener(filepath, mode)
    try: file.extractall()
    finally: file.close()
  finally:
    os.chdir(cwd)


def fetch(FILE, URL, CHECKSUM):
	
  # download to FCNNROOT/data
  filepath = op.realpath(op.join(op.dirname(op.realpath(__file__)), '..', FILE))
  if op.exists(filepath):
    print 'File already downloaded'
  else:
    print 'Downloading models from url: %s \n  to %s' % (URL, filepath)
    f = urllib.URLopener()
    f.retrieve(URL, filepath)
  
  file_is_ok = verify_checksum(filepath, CHECKSUM)

  # extract to FCNNROOT/data
  extracted_path = op.splitext(filepath)[0]
  if op.exists(extracted_path):
    print 'The archive is already unpacked'
  else:
    print 'Extracting archive to %s' % extracted_path
    extract_file(filepath, op.dirname(extracted_path))

  print 'Complete.'
