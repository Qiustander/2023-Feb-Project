import tensorflow as tf
import numpy as np
from typing import Union


def _log_step_initializer(
    tensor: tf.Tensor,  # values should be from U(0, 1)
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> tf.Tensor:
    scale = np.log(dt_max) - np.log(dt_min)
    return tensor * scale + np.log(dt_min)


def _make_omega_l(l_max: int, dtype: tf.dtypes.DType = tf.complex64) -> tf.Tensor:
    return tf.math.exp(tf.cast(tf.range(l_max), dtype=dtype)*2j * np.pi / l_max)


def _make_hippo(N: int) -> np.ndarray:
    def idx2value(n: int, k: int) -> Union[int, float]:
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    hippo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hippo[i, j] = idx2value(i + 1, j + 1)
    return hippo


def _make_nplr_hippo(N: int) :
    nhippo = -1 * _make_hippo(N)

    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]

    lambda_, V = np.linalg.eig(S)
    return lambda_, p, q, V


def _make_p_q_lambda(n: int):
    lambda_, p, q, V = _make_nplr_hippo(n)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    return [tf.convert_to_tensor(i) for i in (p, q, lambda_)]


def _cauchy_dot(v: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    ''' Slow implementation of Cauchy dot
    Args:
        v:
        denominator:
    return:
    '''
    if v._rank() == 1:
        v = v[None, None]
    elif v._rank() == 2:
        v = v[:, None]
    elif v._rank() != 3:
        raise IndexError(f"Expected `v` to be 1D, 2D or 3D, got {v._rank()}D")
    return tf.reduce_sum(v / denominator, axis=-1)


def _non_circular_convolution(u: tf.Tensor, K: tf.Tensor) -> tf.Tensor:
    l_max = u.shape[1] # Need to check the input shape
    # tf.signal.rfft only for the inner-most dimension. Thus we need to transpose.
    ud = tf.signal.rfft(tf.pad(tf.transpose(u, perm=[0, 2, 1]),
                               paddings=tf.constant([[0,0],[0,0],[0,l_max]]))) # [batch, length, feature]
    Kd = tf.signal.rfft(tf.pad(K,
                               paddings=tf.constant([[0,0],[0,0],[0,l_max]])))# [batch, feature, length]
    return tf.cast(tf.transpose(tf.signal.irfft(ud*Kd)[..., :l_max], perm=[0,2,1]), dtype=u.dtype)


def _as_real(x: tf.Tensor) -> tf.Tensor:
    '''
    Args:
        x (tf.Complex64): complex tensor [B, L]
    Return:
        y (tf.Float32): float tensor with real and imaginary parts stacked [B, L, 2]
    '''
    x = x[:, None]
    return tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)


def _as_complex(x: tf.Tensor or tf.Variable) -> tf.Tensor or tf.Variable:
    '''
    Args:
        x (tf.Float32): float tensor with real and imaginary parts stacked [B, L, 2]
    Return:
        y (tf.Complex64): complex tensor [B, L]
    '''
    return tf.complex(x[..., 0], x[..., 1])


class S4Layer(tf.keras.Model):
    """S4 Layer.
    Structured State Space for (Long) Sequences (S4) layer.
    """

    def __init__(self,
                 d_model: int,
                 d_state: int = 64,
                 l_max: int = 1,
                 bidirectional = False,
                 **kernel_args,
                 ):
        '''Initializer.
        Args:
            d_model (int): number of internal features
            d_state (int): dimensionality of the state representation
            l_max (int): length of input signal
            bidirectional (bool): whether to use bidirectional S4
        '''
        super().__init__()
        self.d_model = d_model
        self.n = d_state
        self.l_max = l_max

        p, q, lambda_ = map(lambda t: tf.cast(t, dtype=tf.complex64), _make_p_q_lambda(d_state))
        self._p = tf.Variable(_as_real(p))
        self._q = tf.Variable(_as_real(q))
        self._lambda_ = tf.Variable(_as_real(lambda_))[None, None]

        self.omega_l = _make_omega_l(self.l_max, dtype=tf.complex64)
        initializer = tf.keras.initializers.GlorotNormal()
        self._B = tf.Variable(
            _as_real(tf.complex(initializer(shape=(d_model, d_state)), initializer(shape=(d_model, d_state))))
        )
        self._Ct = tf.Variable(
            _as_real(tf.complex(initializer(shape=(d_model, d_state)), initializer(shape=(d_model, d_state))))
        )
        self.D = tf.Variable(tf.ones((1, 1, d_model)))
        self.log_step = tf.Variable(_log_step_initializer(tf.random.uniform(shape=[d_model])))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n={self.n}, l_max={self.l_max}"

    @property
    def p(self) -> tf.Tensor:
        return _as_complex(self._p)

    @property
    def q(self) -> tf.Tensor:
        return _as_complex(self._q)

    @property
    def lambda_(self) -> tf.Tensor:
        return _as_complex(self._lambda_)

    @property
    def B(self) -> tf.Variable:
        return _as_complex(self._B)

    @property
    def Ct(self) -> tf.Variable:
        return _as_complex(self._Ct)

    def _compute_roots(self) -> tf.Tensor:
        a0, a1 = tf.math.conj(self.Ct), tf.math.conj(self.q)
        b0, b1 = self.B, self.p
        step = tf.exp(self.log_step)

        g = tf.tensordot(tf.cast(2.0 / step, dtype=a0.dtype), (1.0 - self.omega_l) / (1.0 + self.omega_l), 0)
        c = 2.0 / (1.0 + self.omega_l)
        cauchy_dot_denominator = g[..., None] - self.lambda_

        k00 = _cauchy_dot(a0 * b0, denominator=cauchy_dot_denominator)
        k01 = _cauchy_dot(a0 * b1, denominator=cauchy_dot_denominator)
        k10 = _cauchy_dot(a1 * b0, denominator=cauchy_dot_denominator)
        k11 = _cauchy_dot(a1 * b1, denominator=cauchy_dot_denominator)
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    @property
    def K(self) -> tf.Tensor:
        """K convolutional filter."""
        at_roots = self._compute_roots()
        out = tf.math.real(tf.roll(tf.reverse(tf.signal.ifft(at_roots), axis=[-1]), shift=[1], axis=[-1]))
        return out[None]

    def call(self, u: tf.Tensor) -> tf.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return _non_circular_convolution(u, K=self.K) + (self.D * u)
