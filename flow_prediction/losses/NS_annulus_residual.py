import numpy as np
import pydsc
import tensorflow as tf

class NS_annulus_residual:
    #mixed derivative kernel from https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781119083405.app1 eq A.10
    _kernels_d2 = np.array([\
                            [[0,1,0],[0,-2,0],[0,1,0]],\
                            [[1,0,-1],[0,0,0],[-1,0,1]],\
                            [[0,0,0],[1,-2,1],[0,0,0]]\
                            ])
    _kernels_d1 = np.array([\
                            [[0,-1,0],[0,0,0],[0,1,0]],\
                            [[0,0,0],[-1,0,1],[0,0,0]]\
                            ])
    def __init__(self, annulusmap, annulus_polar_coords, nu = 1.0, rho = 1.0, norm_ord = 1, reduce_mean = False, momentum_weight = 1.0, continuity_weight = 1.0):

        # dy = dx if dy is None else dy
        # kernel_d2 = np.einsum('i,i...->i...',np.array([dx**-2, 0.25/(dx*dy), dy**2]),self._kernels_d2)
        # kernel_d1 = np.einsum('i,i...->i...',np.array([0.5/dx, 0.5/dy]),self._kernels_d1)

        self.norm_ord = norm_ord
        self.reduce_mean = reduce_mean
        self.nu = nu
        self.rho = rho
        self.annulus_polar_coords = annulus_polar_coords
        self.annulusmap = annulusmap

        momentum_weight = [momentum_weight, momentum_weight] if isinstance(momentum_weight, float) else momentum_weight
        self.resid_weights = tf.constant(momentum_weight + [continuity_weight], dtype=tf.keras.backend.floatx())

        self._w_coords = self._polar_coords_to_w(self.annulus_polar_coords)

        self._compute_w_z_jacobian_hessian()
        self._compute_polar_w_jacobian_hessian()

        kernels_d1 = tf.cast(tf.expand_dims(tf.transpose(tf.convert_to_tensor(self._kernels_d1), [1,2,0]),2), tf.keras.backend.floatx())
        dr = self.annulus_polar_coords[1,0,0] - self.annulus_polar_coords[0,0,0]
        dtheta = self.annulus_polar_coords[0,1,1] - self.annulus_polar_coords[0,0,1]
        spacings = tf.cast(tf.stack([dr, dtheta],0), tf.keras.backend.floatx())
        self.fd_kernels_d1 = tf.einsum('i,...i->...i', 0.5/spacings, kernels_d1)

        kernels_d2 = tf.cast(tf.expand_dims(tf.transpose(tf.convert_to_tensor(self._kernels_d2), [1,2,0]),2), tf.keras.backend.floatx())
        d2_coeffs = tf.cast(tf.stack([dr**-2, 0.25/(dr*dtheta), dtheta**-2]), tf.keras.backend.floatx())
        self.fd_kernels_d2 = tf.einsum('i,...i->...i', d2_coeffs, kernels_d2)

        self.eval_pressure_term = self._make_pressure_term_calculation_function()
        self.eval_advection_term = self._make_advection_term_calculation_function()
        self.eval_diffusion_term = self._make_diffusion_term_calculation_function()


    @staticmethod
    def _polar_coords_to_w(pc):
        return pc[...,0] * np.exp(pc[...,1] * 1j)
        
    def _compute_w_z_jacobian_hessian(self):
        #evaluate 1st and 2nd complex derivatives of the mapping
        #mapping w = f(z) = f(x+iy) = xi(x,y) + i*eta(x,y)
        dwdz = self.annulusmap.dwdz(self._w_coords)
        d2wdz2 = self.annulusmap.d2wdz2(self._w_coords)
        #use the Cauchy-Riemann relations to obtain the gradients of the real and imag parts of the mapping
        #first derivatives
        self.xi_x = np.real(dwdz)
        self.eta_x = np.imag(dwdz)
        self.xi_y = -self.eta_x
        self.eta_y = self.xi_x
        #second derivatives
        self.xi_x_x = np.real(d2wdz2)
        self.eta_x_y = self.xi_x_x
        self.xi_y_y = -self.xi_x_x
        self.eta_x_x = np.imag(d2wdz2)
        self.xi_x_y = -self.eta_x_x
        self.eta_y_y = -self.eta_x_x
        
        
    def _compute_polar_w_jacobian_hessian(self):
        #evaluate Jacobian and Hessians of the w -> polar mapping
        #simple closed form expressions can be derived from r = sqrt(xi^2+eta^2) and theta=atan(eta/xi)
        xi = np.real(self._w_coords)
        eta = np.imag(self._w_coords)

        #jacobian elements
        r = self.annulus_polar_coords[...,0]
        self.r_xi = xi/r
        self.r_eta = eta/r
        self.theta_xi = -eta/(r**2)
        self.theta_eta = xi/(r**2)

        #hessian elements
        self.r_xi_xi, self.r_xi_eta, self.r_eta_eta = (r**(-3)) * np.stack([eta**2, -xi*eta, xi**2],0)
        self.theta_xi_xi, self.theta_xi_eta, self.theta_eta_eta = (r**(-4)) * np.stack([2*xi*eta, eta**2-xi**2, -2*xi*eta], 0)

    def _make_advection_term_calculation_function(self):
        #u-momentum eq: (u_r*(eta_x*r_eta + r_xi*xi_x) + u_theta*(eta_x*theta_eta + theta_xi*xi_x))*u + (u_r*(eta_y*r_eta + r_xi*xi_y) + u_theta*(eta_y*theta_eta + theta_xi*xi_y))*v
        #v-momentum eq: (v_r*(eta_x*r_eta + r_xi*xi_x) + v_theta*(eta_x*theta_eta + theta_xi*xi_x))*u + (v_r*(eta_y*r_eta + r_xi*xi_y) + v_theta*(eta_y*theta_eta + theta_xi*xi_y))*v

        x_dir_coefficients = tf.stack([self.eta_x*self.r_eta + self.r_xi*self.xi_x, self.eta_x*self.theta_eta + self.theta_xi*self.xi_x],0)#[[r_coeff,theta_coeff],nr,ntheta]
        y_dir_coefficients = tf.stack([self.eta_y*self.r_eta + self.r_xi*self.xi_y, self.eta_y*self.theta_eta + self.theta_xi*self.xi_y],0)

        @tf.function
        def eval_advection_term(u,v):
            
            u_grads_polar = tf.nn.conv2d(u, self.fd_kernels_d1, 1, "SAME", "NCHW", 1)#dudr and du/dtheta; [batch,[u_r,u_theta],nr,ntheta]
            umom_u_coeff = tf.expand_dims(tf.einsum('dij,...dij->...ij', x_dir_coefficients, u_grads_polar),0)#dudx
            umom_v_coeff = tf.expand_dims(tf.einsum('dij,...dij->...ij', y_dir_coefficients, u_grads_polar),0)#dudy
            
            v_grads_polar = tf.nn.conv2d(v, self.fd_kernels_d1, 1, "SAME", "NCHW", 1)
            vmom_u_coeff = tf.expand_dims(tf.einsum('dij,...dij->...ij', x_dir_coefficients, v_grads_polar),0)#dvdx
            vmom_v_coeff = tf.expand_dims(tf.einsum('dij,...dij->...ij', y_dir_coefficients, v_grads_polar),0)#dvdy

            u_term = u * umom_u_coeff + v * umom_v_coeff
            v_term = u * vmom_u_coeff + v * vmom_v_coeff

            advection_terms = tf.concat([u_term, v_term],1)[...,1:-1,1:-1]
            
            return advection_terms, u_grads_polar, umom_u_coeff, v_grads_polar, vmom_v_coeff
        
        return eval_advection_term

    def _make_pressure_term_calculation_function(self):
        p_r_x_coefficient = -(self.eta_x * self.r_eta + self.r_xi * self.xi_x)/self.rho
        p_theta_x_coefficient = -(self.eta_x * self.theta_eta + self.theta_xi * self.xi_x)/self.rho
        p_r_y_coefficient = -(self.eta_y * self.r_eta + self.r_xi * self.xi_y)/self.rho
        p_theta_y_coefficient = -(self.eta_y * self.theta_eta + self.theta_xi * self.xi_y)/self.rho
        p_x_coefficients = tf.cast(tf.stack([p_r_x_coefficient, p_theta_x_coefficient], 0), tf.keras.backend.floatx())
        p_y_coefficients = tf.cast(tf.stack([p_r_y_coefficient, p_theta_y_coefficient], 0), tf.keras.backend.floatx())
        p_coefficients = tf.stack([p_x_coefficients,p_y_coefficients],0)
        
        fd_kernels = self.fd_kernels_d1
        @tf.function
        def eval_pressure_term(p):
            #p_term: shape [batch_size, 2, nx-2, ny-2]. p_term[i,j] contains (-1/rho)*(dp/dx_j)
            pressure_grads_polar = tf.nn.conv2d(p, fd_kernels, 1, "SAME", "NCHW", 1)

            p_term = tf.einsum('...pij,cpij->...cij', pressure_grads_polar, p_coefficients)[...,1:-1,1:-1]

            return p_term

        return eval_pressure_term

    def _make_diffusion_term_calculation_function(self):

        r_xx_coefficient = (self.eta_x*(self.eta_x*self.r_eta_eta + self.r_xi_eta*self.xi_x) + self.eta_x_x*self.r_eta + self.r_xi*self.xi_x_x + self.xi_x*(self.eta_x*self.r_xi_eta + self.r_xi_xi*self.xi_x))
        theta_xx_coefficient = (self.eta_x*(self.eta_x*self.theta_eta_eta + self.theta_xi_eta*self.xi_x) + self.eta_x_x*self.theta_eta + self.theta_xi*self.xi_x_x + self.xi_x*(self.eta_x*self.theta_xi_eta + self.theta_xi_xi*self.xi_x))
        rr_xx_coefficient = (self.eta_x*self.r_eta + self.r_xi*self.xi_x)*(self.eta_x*self.r_eta + self.r_xi*self.xi_x)
        rtheta_xx_coefficient = (self.eta_x*self.r_eta + self.r_xi*self.xi_x)*(self.eta_x*self.theta_eta + self.theta_xi*self.xi_x) + (self.eta_x*self.theta_eta + self.theta_xi*self.xi_x)*(self.eta_x*self.r_eta + self.r_xi*self.xi_x)
        thetatheta_xx_coefficient = (self.eta_x*self.theta_eta + self.theta_xi*self.xi_x)*(self.eta_x*self.theta_eta + self.theta_xi*self.xi_x)
        xx_coefficients = tf.stack([r_xx_coefficient, theta_xx_coefficient, rr_xx_coefficient, rtheta_xx_coefficient, thetatheta_xx_coefficient], 0)

        r_yy_coefficient = (self.eta_y*(self.eta_y*self.r_eta_eta + self.r_xi_eta*self.xi_y) + self.eta_y_y*self.r_eta + self.r_xi*self.xi_y_y + self.xi_y*(self.eta_y*self.r_xi_eta + self.r_xi_xi*self.xi_y))
        theta_yy_coefficient = (self.eta_y*(self.eta_y*self.theta_eta_eta + self.theta_xi_eta*self.xi_y) + self.eta_y_y*self.theta_eta + self.theta_xi*self.xi_y_y + self.xi_y*(self.eta_y*self.theta_xi_eta + self.theta_xi_xi*self.xi_y))
        rr_yy_coefficient = (self.eta_y*self.r_eta + self.r_xi*self.xi_y)*(self.eta_y*self.r_eta + self.r_xi*self.xi_y)
        rtheta_yy_coefficient = (self.eta_y*self.r_eta + self.r_xi*self.xi_y)*(self.eta_y*self.theta_eta + self.theta_xi*self.xi_y) + (self.eta_y*self.theta_eta + self.theta_xi*self.xi_y)*(self.eta_y*self.r_eta + self.r_xi*self.xi_y)
        thetatheta_yy_coefficient = (self.eta_y*self.theta_eta + self.theta_xi*self.xi_y)*(self.eta_y*self.theta_eta + self.theta_xi*self.xi_y)
        yy_coefficients = tf.stack([r_yy_coefficient, theta_yy_coefficient, rr_yy_coefficient, rtheta_yy_coefficient, thetatheta_yy_coefficient], 0)

        @tf.function
        def get_derivs_polar(u, u_grads_polar):
            u_polar_2nd_derivs = tf.nn.conv2d(u, self.fd_kernels_d2, 1, "SAME", "NCHW", 1)
            u_derivs = tf.concat([u_grads_polar, u_polar_2nd_derivs],1)
            return u_derivs

        @tf.function
        def eval_xx_term(u_derivs):
            uxx = tf.einsum('gij,...gij->...ij', xx_coefficients, u_derivs)
            return tf.expand_dims(uxx,1)

        @tf.function
        def eval_yy_term(u_derivs):
            uyy = tf.einsum('gij,...gij->...ij', yy_coefficients, u_derivs)
            return tf.expand_dims(uyy,1)

        @tf.function
        def eval_diffusion_term(u, u_grads_polar, v, v_grads_polar):
            u_derivs = get_derivs_polar(u, u_grads_polar)
            v_derivs = get_derivs_polar(v, v_grads_polar)

            uxx = eval_xx_term(u_derivs)
            uyy = eval_yy_term(u_derivs)
            vxx = eval_xx_term(v_derivs)
            vyy = eval_yy_term(v_derivs)

            diffusion_terms = self.nu * tf.concat([uxx+uyy, vxx+vyy], 1)[...,1:-1,1:-1]

            return diffusion_terms, uxx, uyy, vxx, vyy

        return eval_diffusion_term

    @tf.function
    def __call__(self, u, dudt, v, dvdt, p):
        pterm = self.eval_pressure_term(p)
        aterm,u_grads_polar,ux,v_grads_polar,vy = self.eval_advection_term(u,v)
        diff_term = self.eval_diffusion_term(u, u_grads_polar, v, v_grads_polar)[0]
        acc_term = tf.concat([dudt, dvdt],1)[...,1:-1,1:-1]

        momentum_resid = (acc_term + aterm - pterm - diff_term)
        continuity_resid = (ux+vy)[...,1:-1,1:-1]
        total_resid = tf.concat([momentum_resid, continuity_resid], 1)
        weighted_resid = tf.einsum('bi...,i->bi...', total_resid, self.resid_weights)
        resid_norms = tf.norm(weighted_resid, ord=self.norm_ord, axis=1, keepdims=True)
        
        if self.reduce_mean:
            resid_norms = tf.reduce_mean(resid_norms)

        return resid_norms
