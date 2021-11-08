import string
import tensorflow as tf

def get_attention_tensor_builder(attention_expr):
    varnames, attention_expr = list(map(lambda x: x.strip(), attention_expr.split('|')))
    ndims = len(varnames)

    lambda_expr = 'lambda ' + ','.join(varnames) + ':' + attention_expr
    lambda_fn = eval(lambda_expr)
    
    @tf.function(experimental_relax_shapes=True)
    def build_attention_tensor(tensorshape):
        tensorshape_per_dim = 1-tf.eye(ndims, dtype=tf.int32) + tf.cast(tf.linalg.diag(tensorshape), dtype=tf.int32)
        linspaces = []
        for dim in range(ndims):
            ls = tf.linspace(0.0,1.0,tensorshape[dim])
            ls = tf.reshape(ls, tensorshape_per_dim[dim])
            linspaces.append(
                ls
            )
        att_tensor = lambda_fn(*linspaces)
        return att_tensor
    
    return build_attention_tensor
    
        

class AttentionMeanLpLoss(tf.keras.losses.Loss):
    def __init__(self, attention_expr, p=1, input_shape = None, **kwargs):
        self.p = p
        self._attention_tensor_builder = get_attention_tensor_builder(attention_expr)

        #if input_shape is not None:
        #    self._cached_attention_tensor = self._attention_tensor_builder(tf.zeros(input_shape, dtype=tf.keras.backend.floatx()))
        #else:
        #    self._cached_attention_tensor = None

        super().__init__(**kwargs)

    def call(self, yt, yp):
        yshape = tf.shape(yt)

        # if (self._cached_attention_tensor is None) or (not tf.reduce_all(yshape == tf.shape(self._cached_attention_tensor))):
        #     self._cached_attention_tensor = self._attention_tensor_builder(yshape)

        # loss_val = self._cached_attention_tensor * (tf.abs(yt-yp)**self.p)

        att_tensor = self._attention_tensor_builder(yshape)
        loss_val = att_tensor * (tf.abs(yt-yp)**self.p)
        
        return tf.reduce_mean(loss_val)

if __name__ == '__main__':
    attention_expr = 'bcij | i+j'

    y = tf.random.uniform((10,1,20,20))
    yt = y+1

    loss_fn = AttentionMeanLpLoss(attention_expr, p=1)
    loss_val = loss_fn(yt,y)
    print(loss_val)

        
