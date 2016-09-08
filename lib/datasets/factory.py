# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.flickrlogo32 import flickrlogo32
from datasets.flickrlogo1 import flickrlogo1, binclasses
from datasets.vehicle import vehicle
import numpy as np

# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

for split in ['train', 'test', 'test_try', 'val']:
    name = 'flickrlogo32_{}'.format(split)
    __sets[name] = (lambda split=split: flickrlogo32(split))

for split in ['train', 'test', 'test_try', 'val']:
    for cls in binclasses:
        name = 'flickrlogo1_{}_{}'.format(cls, split)
        __sets[name] = (lambda split=split, cls=cls: flickrlogo1(split, cls))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()



def construct_imdb(name, image_set, data_path):
    if name == 'vehicle':
        return vehicle(image_set=image_set, data_path=data_path)
    elif name == 'flickrlogo32':
        return flickrlogo32(image_set=image_set, data_path=data_path)
