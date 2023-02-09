import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        p, q, r = A.shape
        k = self.upsampling_factor
        upsampled_size = (r * k) - (k - 1)
        Z = np.zeros((p, q, upsampled_size))
        Z[:, :, 0:upsampled_size:k] = A

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        p, q, r = dLdZ.shape
        k = self.upsampling_factor
        downsampled_size = int((r + (k - 1)) / k)
        dLdA = np.zeros((p, q, downsampled_size))  # TODO
        dLdA = dLdZ[:, :, 0:k * r:k]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        p, q, r = A.shape
        self.input_len = r
        k = self.downsampling_factor
        downsampled_size = int((r + (k - 1)) / k)
        Z = np.zeros((p, q, downsampled_size))
        Z = A[:, :, 0:k * r:k]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        p, q, r = dLdZ.shape
        k = self.downsampling_factor
        dLdA = np.zeros((p, q, self.input_len))
        dLdA[:, :, 0:self.input_len:k] = dLdZ


        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        p, q, m, n = A.shape
        k = self.upsampling_factor
        m_new = (m * k) - (k - 1)
        n_new = m_new
        Z = np.zeros((p, q, m_new, n_new))


        for x in range(p):
            for y in range(q):
                j = 0
                for i in range(m):
                    Z[x, y][j][0:m_new:k] = A[x, y][i]
                    j = j + k

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        p, q, m, n = dLdZ.shape
        k = self.upsampling_factor
        downsampled_size = int((m + (k - 1)) / k)
        dLdA = np.zeros((p, q, downsampled_size, downsampled_size))
        for x in range(p):
            for y in range(q):
                j = 0
                for i in range(downsampled_size):
                    dLdA[x, y][i] = dLdZ[x, y][j][0:m:k]
                    j = j + k

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """
        p, q, m, n = A.shape
        self.input_size = m
        k = self.downsampling_factor
        downsampled_size = int((m - 1) / k) + 1
        Z = np.zeros((p, q, downsampled_size, downsampled_size))
        for x in range(p):
            for y in range(q):
                j = 0
                for i in range(downsampled_size):
                    Z[x, y][i] = A[x, y][j][0:m:k]
                    j = j + k

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        p, q, m, n = dLdZ.shape
        k = self.downsampling_factor
        m_new = self.input_size
        n_new = m_new
        dLdA = np.zeros((p, q, m_new, n_new))

        for x in range(p):
            for y in range(q):
                j = 0
                for i in range(m):
                    dLdA[x, y][j][0:m_new:k] = dLdZ[x, y][i]
                    j = j + k

        return dLdA