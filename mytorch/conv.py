# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)

            W: (out_channels, in_channels, kernel_size)
        """

        k = self.kernel_size
        batch_size, _, input_size = A.shape
        output_size = (input_size - k) + 1
        Z = np.zeros((batch_size, self.out_channels, output_size))
        self.A = A
        self.input_size = input_size
        for image in range(batch_size):
            for cOut in range(self.out_channels):
                for i in range(output_size):
                    Z[image, cOut, i] = (A[image, :, i : i + k] * self.W[cOut, :, :]).sum()
                Z[image, cOut] += self.b[cOut]

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)

            dLdW : (out_channels, in_channels, kernel_size)

            A: (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        for batch in range(batch_size):
            for filter in range(out_channels):
                # for cin in range(self.in_channels):
                for i in range(self.kernel_size):
                    # (batch_size, out_channels, output_size)
                    self.dLdW[filter, :, i] += np.sum(dLdZ[batch, filter, :] * self.A[batch, :, i:i+output_size], axis=1)
                    # self.dLdW[filter, cin, i] += np.dot(dLdZ[batch, filter, :], self.A[batch, cin, i:i + output_size])

        # Calculate db
        self.dLdb = np.sum(dLdZ, axis=(0,2))
        k = self.kernel_size


        dLdA = np.zeros(self.A.shape)

        batch_size, in_channels, input_size = self.A.shape
        for batch in range(batch_size):
            # for cin in range(in_channels):
            for i in range(input_size):
                for j in range(out_channels):
                    dz_padded = np.pad(dLdZ[batch, j, :], pad_width=self.kernel_size-1, mode='constant', constant_values=0)
                    flipped_filter = self.W[j,:, ::-1]
                    dLdA[batch, :, i] += np.sum(flipped_filter * dz_padded[i:i+k], axis=1)

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size)
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z_init = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z_init) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        k = self.kernel_size
        batch_size, _, input_width, input_height = A.shape
        output_width = (input_width - k) + 1
        output_height = (input_height - k) + 1
        Z = np.zeros((batch_size, self.out_channels, output_width, output_height))
        self.A = A
        self.input_width = input_width
        self.input_height = input_height
        # for image in range(batch_size):
        for cOut in range(self.out_channels):
            for j in range(output_height):
                for i in range(output_width):
                    Z[:, cOut, i, j] = np.sum((A[:, :, i: i + k, j:j+k ] * self.W[cOut, :, :, :]), axis=(1,2,3))
            Z[:, cOut] += self.b[cOut]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        # for batch in range(batch_size):

            # for cin in range(self.in_channels):
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for filter in range(out_channels):
                # (batch_size, out_channels, output_size)
                    self.dLdW[filter, :, i, j] = np.sum( self.A[:, :, i:i + output_width, j:j+output_height] * dLdZ[:, filter, :, :].reshape(batch_size,1,output_width,output_height) , axis=(0,2,3))

        # Calculate db
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        k = self.kernel_size

        dLdA = np.zeros(self.A.shape)
        # (out_channels, in_channels, kernel_size, kernel_size)
        batch_size, in_channels, input_width, input_height = self.A.shape
        # for batch in range(batch_size):
        for j in range(out_channels):
            dz_padded = np.pad(dLdZ[:, j, :, :], pad_width=((0, 0), (k - 1, k - 1),
                                                            (k - 1, k - 1)),
                               mode='constant', constant_values=0)
            flipped_filter = np.flip(self.W[j, :, :, :], axis=(1, 2))

            for i in range(input_width):
                for l in range(input_height):
                    for cin in range(in_channels):
                        dLdA[:, cin, i, l] += np.sum((dz_padded[:, i:i+k, l:l+k] * flipped_filter[cin, :, :]), axis=(1,2))

        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels,kernel_size) # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z_init = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z_init)


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(self.upsampling_factor) #TODO
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A) #TODO

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)  #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO

        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample1d.backward(delta_out)  #TODO

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) #TODO
        self.upsample2d = Upsample2d(self.upsampling_factor) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A) #TODO

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled) #TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ) #TODO

        dLdA = self.upsample2d.backward(delta_out) #TODO

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.A = A
        self.batch_size, self.in_channels, self.in_width = self.A.shape
        Z = A.reshape(A.shape[0], -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        # dLdA = dLdZ.reshape(dLdZ.shape[0], -1)
        dLdA = dLdZ.reshape(dLdZ.shape[0], self.in_channels, self.in_width)

        return dLdA

