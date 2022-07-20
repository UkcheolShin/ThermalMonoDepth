from __future__ import division
import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics):
        for t in self.transforms:
            images, intrinsics = t(images, intrinsics)
        return images, intrinsics

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, max_value=255):
        self.max_value = max_value

    def __call__(self, images, intrinsics):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float()/self.max_value)
        return tensors, intrinsics


class ArrayToTensorWithLocalWindow(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __init__(self, trust_ratio=1.00):
        self.max_ratio = trust_ratio
        self.min_ratio = 1.0-trust_ratio

    def __call__(self, images, intrinsics):
        tensors = []
        # Decide min-max values of local image window
        tmin = 0
        tmax = 0
        for im in images : 
            im = im.squeeze() #HW
            im_srt = np.sort(im.reshape(-1))
            tmax += im_srt[round(len(im_srt)*self.max_ratio)-1]
            tmin += im_srt[round(len(im_srt)*self.min_ratio)]
        tmax /= len(images)
        tmin /= len(images)

        for im in images:
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            im = torch.clamp(torch.from_numpy(im).float(), tmin, tmax)
            tensors.append((im - tmin)/(tmax - tmin)) #CHW
        return tensors, intrinsics

class TensorThermalRearrange(object):
    def __init__(self, bin_num = 30, CLAHE_clip = 2, CLAHE_tilesize = 8):
        self.bins = bin_num
        self.CLAHE = cv2.createCLAHE(clipLimit=CLAHE_clip, tileGridSize=(CLAHE_tilesize,CLAHE_tilesize))
    def __call__(self, images, intrinsics):
        imgs = []

        tmp_img = torch.cat(images, axis=0)
        hist = torch.histc(tmp_img, bins=self.bins)
        imgs_max = tmp_img.max()
        imgs_min = tmp_img.min()
        itv = (imgs_max - imgs_min)/self.bins
        total_num = hist.sum() 

        for im in images : #CHW
            _,H,W = im.shape
            mul_mask_ = torch.zeros((self.bins,H,W))
            sub_mask_ = torch.zeros((self.bins,H,W))
            subhist_new_min = imgs_min.clone()

            for x in range(0,self.bins) : 
                subhist = (im > imgs_min+itv*x) & (im <= imgs_min+itv*(x+1))
                if (subhist.sum() == 0):
                    continue
                subhist_new_itv = hist[x]/total_num        
                mul_mask_[x,...] = subhist * (subhist_new_itv / itv) 
                sub_mask_[x,...] = subhist * (subhist_new_itv / itv * -(imgs_min+itv*x) + subhist_new_min) 
                subhist_new_min += subhist_new_itv

            mul_mask = mul_mask_.sum(axis=0, keepdim=True).detach()
            sub_mask = sub_mask_.sum(axis=0, keepdim=True).detach()
            im_ = mul_mask*im + sub_mask

            im_ = self.CLAHE.apply((im_.squeeze()*255).numpy().astype(np.uint8)).astype(np.float32)
            im_ = np.expand_dims(im_, axis=2)
            img_out = torch.from_numpy(np.transpose(im_/255., (2, 0, 1)))
            imgs.append(img_out) #CHW
        return imgs, intrinsics

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        if random.random() < 0.5:
            output_intrinsics = np.copy(intrinsics)
            output_images = [np.copy(np.fliplr(im)) for im in images]
            w = output_images[0].shape[1]
            output_intrinsics[:, 0, 2] = w - output_intrinsics[:, 0, 2]
        else:
            output_images = images
            output_intrinsics = intrinsics
        return output_images, output_intrinsics

class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __call__(self, images, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, ch = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[:,0] *= x_scaling
        output_intrinsics[:,1] *= y_scaling

        if ch == 1:
            scaled_images = [np.expand_dims(cv2.resize(im, (scaled_w, scaled_h)), axis=2) for im in images]
        else :
            scaled_images = [cv2.resize(im, (scaled_w, scaled_h)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[:, 0, 2] -= offset_x
        output_intrinsics[:, 1, 2] -= offset_y

        return cropped_images, output_intrinsics