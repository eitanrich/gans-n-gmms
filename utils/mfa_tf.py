import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import mfa
import math
import tensorflow as tf
from tensorflow.contrib.framework import list_variables


def init_raw_parms_np(K, d, l):
    N_PI = np.zeros([K], dtype=np.float32)          # The mixing coefficient logits
    N_MU = np.zeros([K, d], dtype=np.float32)       # Mean values
    N_A = np.zeros([K, d, l], dtype=np.float32)     # Scale - rectangular matrix
    N_D = np.zeros([K, d], dtype=np.float32)        # Isotropic noise
    return N_PI, N_MU, N_A, N_D


def init_raw_parms(K, d, l):
    N_PI, N_MU, N_A, N_D = init_raw_parms_np(K, d, l)
    return tf.Variable(N_PI, name='PI'), tf.Variable(N_MU, name='MU'), tf.Variable(N_A, name='A'), tf.Variable(N_D, name='D')


def init_raw_parms_from_gmm(gmm, prefix=''):
    K = len(gmm.components)
    d, l = gmm.components[0]['A'].shape
    N_PI, N_MU, N_A, N_D = init_raw_parms_np(K, d, l)
    for i, c in gmm.components.items():
        N_PI[i] = np.log(c['pi'])
        N_MU[i, ...] = c['mu']
        N_A[i, ...] = c['A']
        if 's' in c.keys():
            N_D[i, ...] = np.sqrt(c['s'])
        else:
            N_D[i, ...] = np.sqrt(c['D'])
    return tf.Variable(N_PI, name=prefix+'PI'), tf.Variable(N_MU, name=prefix+'MU'), tf.Variable(N_A, name=prefix+'A'), \
           tf.Variable(N_D, name=prefix+'D')


def raw_to_gmm(PI, MU, A, D, raw_as_log=False):
    K = np.size(PI)
    pi_vals = np.exp(PI) / np.sum(np.exp(PI))
    components = {}
    for i in range(K):
        if raw_as_log:
            components[i] = {'pi': pi_vals[i], 'mu': MU[i, ...], 'A': A[i, ...], 'D': np.exp(-1.0 * D[i])}
        else:
            components[i] = {'pi': pi_vals[i], 'mu': MU[i, ...], 'A': A[i, ...], 'D': np.power(D[i], 2.0)}
    return mfa.MFA(components)


def get_per_components_log_likelihood(X, PI_logits, MU, A, sqrt_D):
    """
    Calculate the data log likelihood for low-rank Gaussian Mixture Model.
    See "Learning a Low-rank GMM" (Eitan Richardson) for details
    """
    K, d, l = A.get_shape().as_list()

    # Force isotropic noise...
    # sqrt_D_iso = tf.ones([K, d]) * tf.reshape(tf.reduce_mean(sqrt_D, axis=1), [K, 1])

    # Shapes: A[K, d, l], AT[K, l, d], iD[K, d, 1], L[K, l, l]
    AT = tf.transpose(A, perm=[0, 2, 1])
    iD = tf.reshape(tf.pow(tf.clip_by_value(sqrt_D, 1e-3, 1.0), -2.0), [K, d, 1])
    L = tf.eye(l, batch_shape=[K]) + tf.matmul(AT, iD*A)
    iL = tf.cast(tf.matrix_inverse(tf.cast(L, tf.float64)), tf.float32)

    # Collect the Mahalanibis distances (MD) per sample and component in a tensor array
    m_d_a = tf.TensorArray(dtype=X.dtype, size=K)

    # Each loop iteration calculates the Mahalanobis Distance per component
    def calc_component_m_d(i, ta):
        # Shapes: X_c[d, m], m_d_1[d, m], m_d[m]
        X_c = tf.transpose(X - tf.reshape(MU[i], [1, d]))
        m_d_1 = (iD[i] * X_c) - ((iD[i] * A[i]) @ iL[i]) @ (AT[i] @ (iD[i] * X_c))
        m_d = tf.reduce_sum(X_c * m_d_1, axis=0)
        return i+1, ta.write(i, m_d)

    i = tf.constant(0)
    c = lambda i, _: i < K
    _, final_ta = tf.while_loop(c, calc_component_m_d, loop_vars=(i, m_d_a), swap_memory=True,
                                parallel_iterations=1)
    m_d = final_ta.stack()

    # Shapes: m_d[K, m], log_det_Sigma[K], component_log_probs[K, 1], log_prob_data_given_components[K, m]
    # log_det_Sigma = tf.log(tf.matrix_determinant(L)) - tf.reduce_sum(minus_log_D, axis=1)
    det_L = tf.log(tf.matrix_determinant(tf.cast(L, tf.float64)))
    log_det_Sigma = tf.cast(det_L, tf.float32) - tf.reduce_sum(tf.log(tf.reshape(iD, [K, d])), axis=1)
    # log_det_Sigma = tf.log(tf.matrix_determinant(L)) - tf.reduce_sum(tf.log(tf.reshape(iD, [K, d])), axis=1)
    log_prob_data_given_components = -0.5 * (tf.reshape(d*tf.log(2.0*math.pi) + log_det_Sigma, [K, 1]) + m_d)
    component_log_probs = tf.reshape(tf.log(tf.nn.softmax(PI_logits)), [K, 1])
    return component_log_probs + log_prob_data_given_components


def get_log_likelihood(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    LLs = tf.reduce_logsumexp(comp_LLs, axis=0)
    return tf.reduce_sum(LLs)


def get_per_sample_log_likelihood(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    LLs = tf.reduce_logsumexp(comp_LLs, axis=0)
    return LLs


def get_per_sample_per_component_log_prob(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    return comp_LLs


def get_per_sample_log_responsibilities(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    return comp_LLs - tf.reduce_logsumexp(comp_LLs, axis=0)


def get_per_sample_responsibilities(X, PI, MU, A, D):
    return tf.exp(get_per_sample_log_responsibilities(X, PI, MU, A, D))


def get_max_posterior_component(X, PI, MU, A, D):
    comp_LLs = get_per_components_log_likelihood(X, PI, MU, A, D)
    return tf.argmax(comp_LLs, axis=0)


def get_latent_posterior_mean(X, c_i, MU, A, sqrt_D):
    """
    Calculate the posterior probability of the latent variable z given x, for selected Gaussians
    """
    K, d, l = A.get_shape().as_list()

    # Shapes: A[K, d, l], AT[K, l, d], iD[K, d, 1], L[K, l, l]
    AT = tf.transpose(A, perm=[0, 2, 1])
    iD = tf.reshape(tf.pow(tf.clip_by_value(sqrt_D, 1e-3, 1.0), -2.0), [K, d, 1])
    L = tf.eye(l, batch_shape=[K]) + tf.matmul(AT, iD*A)
    iL = tf.cast(tf.matrix_inverse(tf.cast(L, tf.float64)), tf.float32)

    # Shapes: X_c[m, d, 1], iD_c[m, d, 1]
    X_c = tf.reshape(X - tf.gather(MU, c_i), [-1, d, 1])
    iD_c = tf.reshape(tf.gather(iD, c_i), [-1, d, 1])
    m_d_1 = (iD_c * X_c) - ((iD_c * tf.gather(A, c_i)) @ tf.gather(iL, c_i)) @ (tf.gather(AT, c_i) @ (iD_c * X_c))
    mu_z = tf.gather(AT, c_i) @ m_d_1
    return mu_z


def generate_from_posterior(X, PI_logits, MU, A, sqrt_D):
    K, d, l = A.get_shape().as_list()

    # Find the most probable components for each sample
    c_i = tf.reshape(get_max_posterior_component(X, PI_logits, MU, A, sqrt_D), [-1])

    # Calculate the posterior mean z values and use them to generate samples from the model
    z_mu_i = get_latent_posterior_mean(X, c_i, MU, A, sqrt_D)

    # Generate new samples from the posterior mean (ignoring the posterior covariance for now...)
    return tf.gather(A, c_i) @ z_mu_i + tf.reshape(tf.gather(MU, c_i), [-1, d, 1])

