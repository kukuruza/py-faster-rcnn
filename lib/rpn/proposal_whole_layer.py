import caffe
import numpy as np
import yaml

DEBUG = False

class ProposalWholeLayer(caffe.Layer):
    """
    Outputs identical proposals, each proposal being the whole image.
    Size of proposals (whole image size) is taken from bottom[0]=image blob,
    number of proposals is taken to match the number of bottom[1]=other_proposals
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        # bottom[0] original image (width, height)
        # bottom[1] proposals blob for is used for the number of proposals

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)

    def forward(self, bottom, top):

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        assert len(bottom[1].data.shape) == 2 and bottom[1].data.shape[1] == 5, \
            'bottom[1] must be of shape (N,5)'

        im_info = bottom[0].data[0, :]
        width  = im_info[0]
        height = im_info[1]

        number = bottom[1].data.shape[0]

        if DEBUG:
            print 'im_size: (%d, %d)' % (width, height)
            print 'number of proposals: %d' % number

        # Output rois blob
        # Our RPN implementation only supports a single input image, so all
        # batch inds are 0
        blob = np.repeat (np.asarray([[0,0,0,width,height]], np.float32), 
            repeats=number, axis=0)

        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

