import cv2 # type: ignore
import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Fusion:
    def __init__(self, input, model):
        """
        Class Fusion constructor

        Instance Variables:
            self.images: input images
            self.model: CNN model, default=vgg19
            self.device: either 'cuda' or 'cpu'
        """
        self.input_images = input
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.normalized_images = [-1 for img in self.input_images]
        self.YCbCr_images = [-1 for img in self.input_images]
        self.images_to_tensors = []
        
    def fuse(self):
        """
        A top level method which fuse self.images
        """
        #
        # Trasforma le immagini a valori interi in immagini con valori
        # Float compresi fra 0 e 1.
        # Nel caso di immagini RGB, viene applicata la conversione al formato YCbCr
        self.normalize()
        #        
        # Transfer all images to PyTorch tensors
        self._tranfer_to_tensor()
        # Perform fuse strategy
        #
        #
        fused_img = self._fuse()[:, :, 0]
        #
        #
        # Reconstruct fused image given rgb input images
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx][:, :, 0] = fused_img
                fused_img = self._YCbCr_to_RGB(self.YCbCr_images[idx])
                #
                # Given an interval, values outside the interval are clipped to the interval edges.
                # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
                #
                fused_img = np.clip(fused_img, 0, 1)

        return (fused_img * 255).astype(np.uint8)

    def _fuse(self):
        """
        Perform fusion algorithm
        """
        with torch.no_grad():

            imgs_sum_maps = [-1 for tensor_img in self.images_to_tensors]
            #
            # Per ogni immagine passata in input
            #
            for idx, tensor_img in enumerate(self.images_to_tensors):
                imgs_sum_maps[idx] = []

                feature_maps = self.model(tensor_img)

                for feature_map in feature_maps:
                    #plt.imshow(feature_map[0][0])
                    #
                    # ciascun la dimensione di ciascuna feature_map è (1, 64, 256, 256)
                    # dove 64 è il numero di kernel utilizzati. ci sarà una immagine che rappresenta
                    # le features estratte dall'ìesimo operatore convoluzionale.
                    # Vengono sommate per ottenere una singola immagine che contiene tutte le features!!
                    #
                    # torch.sum()
                    # Returns the sum of each row of the input tensor in the given dimension dim. 
                    # If dim is a list of dimensions, reduce over all of them.
                    # If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1.
                    # Otherwise, dim is squeezed (see torch.squeeze()), resulting in the output tensor having 1 (or len(dim)) fewer dimension(s).
                    #
                    sum_map = torch.sum(feature_map, dim=1, keepdim=True)
                    imgs_sum_maps[idx].append(sum_map)
                #plt.imshow(imgs_sum_maps[idx])
            max_fusion = None
            for sum_maps in zip(*imgs_sum_maps):

                self.features = torch.cat(sum_maps, dim=1)
                #
                # SIZE_OF_IMG = self.images_to_tensors[0].shape[2:]. Nel caso di immagini prese da brain atlas 256X256
                #
                weights = self._softmax(F.interpolate(self.features,
                                        size=self.images_to_tensors[0].shape[2:]))
                #
                # F.interpolate()
                # Down/up samples the input to either the given size or the given scale_factor
                # The algorithm used for interpolation is determined by mode.
                # mode='nearest' by default
                #
                self.weights = F.interpolate(weights,
                                        size=self.images_to_tensors[0].shape[2:])

                current_fusion = torch.zeros(self.images_to_tensors[0].shape)
                
                for idx, tensor_img in enumerate(self.images_to_tensors):
                    #
                    # Pesano il contributo di ogni pixel
                    #
                    current_fusion += tensor_img * weights[:,idx]
                if max_fusion is None:
                    max_fusion = current_fusion
                else:
                    max_fusion = torch.max(max_fusion, current_fusion)

            output = np.squeeze(max_fusion.cpu().numpy())
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            return output

    def _RGB_to_YCbCr(self, img_RGB):
        """
        A private method which converts an RGB image to YCrCb format
        """
        img_RGB = img_RGB.astype(np.float32) / 255.
        return cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    def _YCbCr_to_RGB(self, img_YCbCr):
        """
        A private method which converts a YCrCb image to RGB format
        """
        img_YCbCr = img_YCbCr.astype(np.float32)
        return cv2.cvtColor(img_YCbCr, cv2.COLOR_YCrCb2RGB)

    def _is_gray(self, img):
        """
        A private method which returns True if image is gray, otherwise False
        """
        if len(img.shape) < 3:
            return True
        if img.shape[2] == 1:
            return True
        b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
        if (b == g).all() and (b == r).all():
            return True
        return False

    def _softmax(self, tensor):
        """
        A private method which compute softmax ouput of a given tensor
        """
        tensor = torch.exp(tensor)
        tensor = tensor / tensor.sum(dim=1, keepdim=True)
        return tensor

    def _tranfer_to_tensor(self):
        """
        A private method to transfer all input images to PyTorch tensors
        """
        for image in self.normalized_images:
            np_input = image.astype(np.float32)
            if np_input.ndim == 2:
                np_input = np.repeat(np_input[None, None], 3, axis=1)
            else:
                np_input = np.transpose(np_input, (2, 0, 1))[None]
            if self.device == "cuda":
                self.images_to_tensors.append(torch.from_numpy(np_input).cuda())
            else:
                self.images_to_tensors.append(torch.from_numpy(np_input))

    def normalize(self):
        """
         Convert all images to YCbCr format
        """
        for idx, img in enumerate(self.input_images):
            if not self._is_gray(img):
                self.YCbCr_images[idx] = self._RGB_to_YCbCr(img)
                self.normalized_images[idx] = self.YCbCr_images[idx][:, :, 0]
            else:
                self.normalized_images[idx] = img / 255.
