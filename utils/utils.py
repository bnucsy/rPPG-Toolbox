""" Short functions for data-preprocessing and data-loading. """

import numpy as np
import cv2
import torch


def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def sample(a, len):
    """Samples a sequence into specific length."""
    return np.interp(
        np.linspace(
            1, a.shape[0], len), np.linspace(
            1, a.shape[0], a.shape[0]), a)

# feature  -->   [ batch, patch*patch, channel, T]
# torch.Size([1, 64, 64, 60, 60])
def FeatureMap2Heatmap(x : torch.Tensor, Score1, Score2, Score3, log_dir):
    '''
    x: [B, C, T, H, W]
    Score1: [B, head, H, W]
    Score2: [B, head, H, W]
    Score3: [B, head, H, W]
    log_dir: str
    this function is used to save the feature map to the log_dir
    the save dir is :
    org_img: log_dir/log_dir/visual.jpg
    Score[123]: log_dir/Score1_head1.jpg

    note : 因为此函数常用于训练过程中，因此接受 GPU 上的变量
    '''
    ## initial images 
    org_img = x[0,:,32,:,:].cpu()   # RGB
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))

    cv2.imwrite(log_dir + '/' + log_dir + 'visual.jpg', org_img)
 

    # [B, head, 640, 640]
    org_img = Score1[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(log_dir + '/' + 'Score1_head1.jpg', org_img)
    
    
    org_img = Score2[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(log_dir + '/' + 'Score2_head1.jpg', org_img)
    
    
    org_img = Score3[0, 1].cpu().data.numpy() * 4000
    org_img = cv2.cvtColor(org_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(log_dir + '/' + 'Score3_head1.jpg', org_img)

