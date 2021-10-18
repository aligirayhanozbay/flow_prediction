import tensorflow as tf

from .NS_annulus_residual import NS_annulus_residual

class NS_annulus_residual_driver:
    def __init__(self, cases, reduce_mean = True, **loss_opts):
        self.residual_losses = [NS_annulus_residual(reduce_mean = True, **{**loss_opts, **case}) for case in cases]
        self._loss_ids = list(range(len(self.residual_losses)))
        self.reduce_mean = reduce_mean

    #@tf.function
    def _eval_loss_by_case_id(self, args):
        case_id, u, dudt, v, dvdt, p = args
        case_id = tf.cast(case_id, tf.int32)
        branch_fn_closures = list(map(lambda f: (lambda: f(u,dudt,v,dvdt,p)), self.residual_losses))
        branch_fns = dict(zip(self._loss_ids, branch_fn_closures))
        res = tf.switch_case(case_id, branch_fns = branch_fns)
        return res

    def __call__(self, u, dudt, v, dvdt, p, case_ids):
        case_names, case_indices = tf.unique(case_ids)
        n_cases_in_input = tf.shape(case_names)[0]
        
        case_mask = ((tf.expand_dims(case_ids,1) - tf.expand_dims(case_names,0)) == 0)
        n_samples_per_case = tf.reduce_sum(tf.cast(case_mask,tf.int32),0)
        sorted_case_indices = tf.argsort(case_indices)
        
        sorted_u = tf.gather(u, sorted_case_indices)
        sorted_dudt = tf.gather(dudt, sorted_case_indices)
        sorted_v = tf.gather(v, sorted_case_indices)
        sorted_dvdt = tf.gather(dvdt, sorted_case_indices)
        sorted_p = tf.gather(p, sorted_case_indices)
        
        row_splits = tf.concat([[0],tf.cumsum(n_samples_per_case)],0)
        u_by_case = tf.RaggedTensor.from_row_splits(sorted_u, row_splits)
        dudt_by_case = tf.RaggedTensor.from_row_splits(sorted_dudt, row_splits)
        v_by_case = tf.RaggedTensor.from_row_splits(sorted_v, row_splits)
        dvdt_by_case = tf.RaggedTensor.from_row_splits(sorted_dvdt, row_splits)
        p_by_case = tf.RaggedTensor.from_row_splits(sorted_p, row_splits)
        
        losses_by_case = tf.map_fn(self._eval_loss_by_case_id, elems=(case_names, u_by_case, dudt_by_case, v_by_case, dvdt_by_case, p_by_case), fn_output_signature=tf.keras.backend.floatx())

        if self.reduce_mean:
            losses_by_case = tf.reduce_mean(losses_by_case)

        return losses_by_case
        
