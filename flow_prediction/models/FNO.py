import numpy as np
import tensorflow as tf

# Li Z, Kovachki N, Azizzadenesheli K, Liu B, Bhattacharya K, Stuart A, Anandkumar A. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895. 2020 Oct 18.


def complex_uniform_initializer(scale=0.05):
    real_initializer = tf.keras.initializers.RandomUniform(-scale,scale)
    def initializer(shape,dtype):
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = real_initializer(shape,dtype)
        imag = real_initializer(shape,dtype)
        return tf.dtypes.complex(real,imag)
    return initializer

################################################################
# fourier layer
################################################################
class SpectralConv2d(tf.keras.layers.Layer):
    def __init__(self, out_channels, modes1, modes2):
        super().__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.w1r_coords = tf.stack(tf.meshgrid(tf.range(self.modes1), tf.range(self.modes2), indexing='ij'),-1)

    def build(self, input_shape):
        self.in_channels = input_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        initializer = complex_uniform_initializer(self.scale)

        var_shape = (self.in_channels, self.out_channels, self.modes1, self.modes2)

        self.w1 = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.complex64, name='w1')
        self.w2 = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.complex64, name='w2')

    # Complex multiplication
    @tf.function
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return tf.einsum("bixy,ioxy->xybo", input, weights)

    @tf.function
    def call(self, x):
        xshape = tf.shape(x)
        batchsize = xshape[0]
        fft_out_size_dim0 = xshape[2]
        fft_out_size_dim1 = (xshape[3]//2)+1
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = tf.signal.rfft2d(x)

        # Multiply relevant Fourier modes
        w1r = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.w1)
        w2r = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.w2)
        wr = tf.concat([w1r, w2r],0)
        w2r_coords = self.w1r_coords + (fft_out_size_dim0-self.modes1)*tf.constant([[[1,0]]], dtype=self.w1r_coords.dtype)
        wr_coords = tf.concat([self.w1r_coords, w2r_coords],0)
        out_ft = tf.scatter_nd(wr_coords, wr, (fft_out_size_dim0, fft_out_size_dim1, batchsize, self.out_channels))
        out_ft = tf.transpose(out_ft, (2,3,0,1))

        #Return to physical space
        return tf.signal.irfft2d(out_ft, fft_length=xshape[2:])
