"""The data layer used during training to train a Fast R-CNN network.

AngleMapsLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import cv2
import yaml
from multiprocessing import Process, Queue

import sys, os, os.path as op
sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from learning.helperImg import ReaderVideo


class AngleMapsLayer(caffe.Layer):
  """Fast R-CNN data layer used for training."""

  def setup(self, bottom, top):

    map_path = op.join(os.getenv('CITY_DATA_PATH'),
                       "augmentation/scenes/cam572/google1/map_yaw.png")

    self.map_yaw = cv2.imread(map_path, 0)
    assert self.map_yaw is not None

    assert len(bottom) == 1,            'requires a single layer.bottom'
    assert bottom[0].data.ndim == 4,    'requires image data'
    assert len(top) == 1,               'requires a single layer.top'

    # the shape of last layer map is input by the user
    size = cfg.TOP_MAP_SIZE
    top[0].reshape(1, 1, size[0], size[1])



  def forward(self, bottom, top):

    in_blob = bottom[0].data[...]
    size = in_blob.shape
    #print size

    # resize to the size of input
    resized_map = cv2.resize(self.map_yaw, (size[3],size[2]))
    # to float, subtract mean, normalize
    resized_map = (resized_map.astype(np.float32) - 45) / 45
    # supports only a batch of one image
    assert bottom[0].data[...].shape[0] == 1

    resized_map = resized_map[np.newaxis,np.newaxis,:,:]
    assert resized_map.shape[2:4] == in_blob.shape[2:4]
    assert resized_map.shape[0:2] == (1,1)
    
    # assign to top
    top[0].reshape(*(resized_map.shape))
    top[0].data[...] = resized_map

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass

  def reshape(self, bottom, top):
    """Reshaping happens during the call to forward."""
    pass
