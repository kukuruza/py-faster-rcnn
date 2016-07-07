import caffe
import numpy as np
import yaml
from expand_roi import expandRoiFloat

DEBUG = False



class ModifyRoisLayer(caffe.Layer):
    """
    Takes ROIs in the form of on input and modifies them.
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._perc_increase = layer_params['perc_increase']

        # bottom[0] image blob is used for the size of proposals
        # bottom[1] rois = N x [batch_id, x1, y1, x2, y2]

        # top[0] holds N modified rois.

    def forward(self, bottom, top):

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        assert len(bottom[1].data.shape) == 2 and bottom[1].data.shape[1] == 5, \
            'bottom[1] must be of shape (N,5)'

        im_info = bottom[0].data[0, :]
        width  = im_info[0]
        height = im_info[1]

        proposals = bottom[1].data
        # (at the moment only one modification is possible -- increase each roi)
        proposals = proposals[:,1:]  # remove the batch_id col
        
        if DEBUG:
            print 'im_size: (%d, %d)' % (width, height)
            print 'number of rois: %d' % proposals.shape[0]
            print 'rois before modification: \n%s' % str(proposals)

        # modify each proposal 
        for i,proposal in enumerate(proposals):
            proposal = expandRoiFloat(proposal, (width, height), self._perc_increase)
            proposals[i,:] = np.asarray(proposal, dtype=np.float32)

        if DEBUG:
            print 'rois after modification: \n%s' % str(proposals)

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

