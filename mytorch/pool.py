import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A

        k = self.kernel

        batch_size, in_channels, input_width, input_height = A.shape
        out_channels = in_channels
        output_width = (input_width - k) + 1
        output_height = (input_height - k) + 1
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        self.Zind = np.zeros((batch_size, out_channels, 2, output_width, output_height))
        self.A = A
        self.input_width = input_width
        self.input_height = input_height
        for image in range(batch_size):
            for cin in range(in_channels):
                for i in range(output_width):
                    for j in range(output_height):
                        Z[image, cin, i, j] = np.max(self.A[image, cin, i:i+k, j:j+k])
                        # x = np.unravel_index(np.argmax(self.A[image,cin, i:i+k, j:j+k]), (k,k))
                        # self.Zind[image, cin, 0, i, j] = int(i + x[0])
                        # self.Zind[image, cin, 1, i, j] = int(j + x[1])


        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        k = self.kernel

        self.dLdA = np.zeros(self.A.shape)
        # (out_channels, in_channels, kernel_size, kernel_size)
        batch_size, in_channels, input_width, input_height = self.A.shape

        # for q in range(in_channels):
        for batch in range(batch_size):
            for cout in range(out_channels):
                for x in range(output_width):
                    for y in range(output_height):
                        self.dLdA[batch, cout, x:x + k, y:y + k] += dLdZ[batch, cout, x, y] * (self.A[batch, cout, x:x+k, y:y+k] == np.max(self.A[batch, cout, x:x+k, y:y+k]))

        return self.dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A = A

        k = self.kernel

        batch_size, in_channels, input_width, input_height = A.shape
        out_channels = in_channels
        output_width = (input_width - k) + 1
        output_height = (input_height - k) + 1
        Z = np.zeros((batch_size, out_channels, output_width, output_height))
        self.A = A
        self.input_width = input_width
        self.input_height = input_height
        # for image in range(batch_size):
        for cOut in range(out_channels):
            for j in range(output_height):
                for i in range(output_width):
                    Z[:, cOut, i, j] += np.mean(A[:, cOut, i:i + k, j:j + k], axis=(1,2))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        k = self.kernel

        dLdA = np.zeros(self.A.shape)
        # (out_channels, in_channels, kernel_size, kernel_size)
        batch_size, in_channels, input_width, input_height = self.A.shape
        # for batch in range(batch_size):
        # for q in range(in_channels):
        for x in range(output_width):
            for y in range(output_height):
                for i in range(k):
                    for j in range(k):
                        dLdA[:, :, x+i, y+j] += (1/k**2) * dLdZ[:,:,x,y]

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel) #TODO
        self.downsample2d = Downsample2d(self.stride) #TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(Z)

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
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA

