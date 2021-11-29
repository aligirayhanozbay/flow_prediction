import numpy as np
import tensorflow as tf
import itertools

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
class SpectralConv(tf.keras.layers.Layer):
    _fft_funcs = {
        1:{'rfft': tf.signal.rfft, 'irfft': tf.signal.irfft},
        2:{'rfft': tf.signal.rfft2d, 'irfft': tf.signal.irfft2d},
        3:{'rfft': tf.signal.rfft3d, 'irfft': tf.signal.irfft3d}
    }
    _transpose_indices = {
        1:(2,0,1),
        2:(2,3,0,1),
        3:(3,4,0,1,2)
    }
    def __init__(self, out_channels, modes, activation=None):
        super().__init__()
        
        self.out_channels = out_channels
        self.modes = [int(x) for x in modes] #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.ndims = len(self.modes)
        assert (self.ndims > 0) and (self.ndims < 4)

        self._modes_tensor = tf.constant(self.modes, dtype=tf.int32)
        self._wr_slice_sizes = tf.concat([[-1,-1], self._modes_tensor],0)
        self._build_wr_slicing_modes()
        self._image_data_format_transpose_idxs = [0] + [self.ndims+2-1] + list(range(1,1+self.ndims))
        self._image_data_format_rev_transpose_idxs = [0] + list(range(2,2+self.ndims)) + [1]
        
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == 'channels_first':
            self.in_channels = input_shape[1]
        else:
            self.in_channels = input_shape[-1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        initializer = complex_uniform_initializer(self.scale)

        var_shape = (2**(self.ndims-1), self.in_channels, self.out_channels, *self.modes)

        self.w = self.add_weight(shape = var_shape, initializer = initializer, dtype = tf.complex64, name='w')

    # Complex multiplication
    @tf.function
    def compl_mul(self, inp, weights):
        # (fft corner, batch, in_channel, x0, x1, ...), (fft corner, in_channel, out_channel, x0, x1, ...) -> (fft corner, batch, out_channel, x0, x1, ...)
        return tf.einsum("wbi...,wio...->w...bo", inp, weights)

    @staticmethod
    @tf.function
    def _get_fft_out_size(input_shape):
        final_spatial_dim_size = input_shape[-1]
        other_dims_size = input_shape[2:-1]
        return tf.concat([other_dims_size, [(final_spatial_dim_size//2)+1]],0)

    def _build_wr_slicing_modes(self):
        #fft bins are placed at the half of the vertices of an n-dimensional cube.
        n=tf.shape(self._modes_tensor)[0]-1
        i = [0,1]
        modes = np.array(list(itertools.product(*list(itertools.repeat(i,n)))))[:,::-1]
        modes = np.concatenate([modes, np.zeros((modes.shape[0],1), dtype=modes.dtype)],1)
        self._wr_slicing_modes = tf.constant(modes, dtype=tf.int32)


    @tf.function
    def _build_single_scatter_meshgrid(self,starts,ends):
        return tf.stack(tf.meshgrid(*[tf.range(starts[k], ends[k]) for k in range(self.ndims)], indexing='ij'),-1)

    @tf.function
    def _build_scatter_indices(self, starts, ends):
        si = tf.map_fn(lambda x: self._build_single_scatter_meshgrid(x[0], x[1]), (starts, ends), fn_output_signature=tf.int32)
        return si

    @tf.function
    def _wr_slice(self, tensor):
        spatial_dim_sizes = tf.shape(tensor)[2:]
        starts = tf.zeros((self.ndims,), dtype=tf.int32) + self._wr_slicing_modes * (spatial_dim_sizes - self._modes_tensor)
        starts = tf.concat([tf.zeros((2**(self.ndims-1), 2), dtype=tf.int32), starts],1)
        slices = tf.vectorized_map(lambda s: tf.slice(tensor, s, self._wr_slice_sizes), starts)

        scatter_starts = starts[:,2:]
        scatter_ends = (starts + self._wr_slice_sizes)[:,2:]
        scatter_indices = self._build_scatter_indices(scatter_starts, scatter_ends)
        return slices, scatter_indices, spatial_dim_sizes
        
    @tf.function
    def call(self, x):
        if tf.keras.backend.image_data_format() == 'channels_last':
            x = tf.transpose(x, self._image_data_format_transpose_idxs)
        
        xshape = tf.shape(x)
        batchsize = xshape[0]
        fft_out_size = self._get_fft_out_size(xshape)
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = self._fft_funcs[self.ndims]['rfft'](x)

        # Multiply relevant Fourier modes
        wr,scatter_ind, fft_sizes = self._wr_slice(x_ft)
        wr = self.compl_mul(wr, self.w)
        fft_tensor_size = tf.concat([fft_sizes, [batchsize, self.out_channels]],0)
        out_ft = tf.scatter_nd(scatter_ind, wr, fft_tensor_size)
        out_ft = tf.transpose(out_ft, self._transpose_indices[self.ndims])

        #Return to physical space
        out = self._fft_funcs[self.ndims]['irfft'](out_ft, fft_length = xshape[2:])

        if tf.keras.backend.image_data_format() == 'channels_last':
            out = tf.transpose(out, self._image_data_format_rev_transpose_idxs)
            
        return self.activation(out)


if __name__ == '__main__':
    tf.keras.backend.set_image_data_format('channels_first')
    bsize = 10
    nc = 4
    ns = [64,59,37]
    if tf.keras.backend.image_data_format() == 'channels_first':
        inp = tf.random.uniform([bsize, nc, *ns])
    else:
        inp = tf.random.uniform([bsize, *ns, nc])

    lay = SpectralConv(13, [16 for _ in ns])
    print(lay(inp).shape)
