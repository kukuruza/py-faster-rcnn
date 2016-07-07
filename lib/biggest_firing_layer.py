#from __future__ import print_function
import caffe
import yaml
import numpy as np
import json
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps

DEBUG = False
np.set_printoptions(precision=2, linewidth=150)

class BiggestFiringLayer(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._threshold = layer_params['threshold']

        #top[0].reshape(1, 1)

    def forward(self, bottom, top):
        '''
        bottom[0] -- pool5
        bottom[1] -- pool5 tansformed by ST
        bottom[2] -- full_theta
        bottom[3] -- rois
        '''

        data1 = bottom[0].data
        data2 = bottom[1].data
        assert data1.shape[2] == 28 and data1.shape[3] == 28, data1.shape
        assert data2.shape[2] == 7  and data2.shape[3] == 7, data2.shape
        #data1 = scipy.misc.imresize(arr, data2.shape[], interp='bilinear')
        #data = abs(data1 - data2)
        #assert len(data.shape) == 4

        print 'BiggestFiringLayer:'
#        hist_diff, _ = np.histogram(data, range=(0, 10), bins=100)
#        print hist_diff
#        hist_1, _ = np.histogram(data1, range=(0, 10), bins=100)
#        print hist_1
        
        print bottom[2].data
        
#        print bottom[3].data

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


