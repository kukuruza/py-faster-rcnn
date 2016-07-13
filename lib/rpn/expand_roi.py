import numpy as np

DEBUG = False

def expandRoiFloat (roi, (imwidth, imheight), perc):
    '''
    roi - [x1 y1 x2 y2]
    perc - are (forward + backwards) perc.
    floats are rounded to the nearest integer
    '''
    roi0 = roi.copy()
    if perc == 0: return roi

    half_delta_x = float(roi[2] - roi[0]) * perc / 2
    half_delta_y = float(roi[3] - roi[1]) * perc / 2
    # the result must be within (imheight, imwidth)
    bbox_width  = roi[2] - roi[0] + half_delta_x * 2
    bbox_height = roi[3] - roi[1] + half_delta_y * 2
    if bbox_height > imheight or bbox_width > imwidth:
        #print ('expanded bbox of size (%d,%d) does not fit into image (%d,%d)' %
        #    (bbox_height, bbox_width, imheight, imwidth))
        # if so, decrease half_delta_y, half_delta_x
        coef = min(imheight / bbox_height, imwidth / bbox_width)
        bbox_height *= coef
        bbox_width  *= coef
        if DEBUG: print ('decreased bbox to (%d,%d)' % (bbox_width, bbox_height))
        half_delta_x = (bbox_width  - (roi[2] + 1 - roi[0])) * 0.5
        half_delta_y = (bbox_height - (roi[3] + 1 - roi[1])) * 0.5
        #logging.warning ('perc_y, perc_x: %.1f, %1.f: ' % (perc_y, perc_x))
        #logging.warning ('original roi: %s' % str(roi))
        #logging.warning ('half_delta-s y: %.1f, x: %.1f' % (half_delta_y, half_delta_x))
    # and a small epsilon to account for floating-point imprecisions
    EPS = 0.001
    # expand each side
    roi[0] -= (half_delta_x - EPS)
    roi[1] -= (half_delta_y - EPS)
    roi[2] += (half_delta_x - EPS)
    roi[3] += (half_delta_y - EPS)
    # move to clip into borders
    if roi[0] < 0:
        roi[2] += abs(roi[0])
        roi[0] = 0
    if roi[1] < 0:
        roi[3] += abs(roi[1])
        roi[1] = 0
    if roi[2] > imwidth-1:
        roi[0] -= abs((imwidth-1) - roi[2])
        roi[2] = imwidth-1
    if roi[3] > imheight-1:
        roi[1] -= abs((imheight-1) - roi[3])
        roi[3] = imheight-1
    # check that now averything is within borders (bbox is not too big)
    roi = [round(x) for x in roi]
    err_str = 'old: %s, new: %s, imwidth: %d, imheight: %d' % (str(roi0), str(roi), imwidth, imheight)
    return roi



if __name__ == "__main__":
    roi = np.asarray([ 139.0625,   204.6875,   412.5,      350.78125], dtype=np.float32)
    imwidth, imheight = 600, 800
    perc = 1.0
    roi = expandRoiFloat (roi, (imwidth, imheight), perc)
    print roi
