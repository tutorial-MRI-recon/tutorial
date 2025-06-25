import tensorflow as tf
import tf_utils
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class data_consistency():
    """
    Data consistency class can be used for:
        -performing E^h*E operation in the paper
        -transforming final network output to kspace
    """

    def __init__(self, sens_maps, mask):
        with tf.compat.v1.name_scope('EncoderParams'):
            self.shape_list = tf.shape(input=mask)
            self.sens_maps = sens_maps
            self.mask = mask
            self.shape_list = tf.shape(input=mask)
            self.scalar = tf.complex(tf.sqrt(tf.cast(self.shape_list[0] * self.shape_list[1], dtype=tf.float32)), 0.)

    def EhE_Op(self, img, mu):
        """
        Performs (E^h*E+ mu*I) x
        """
        with tf.compat.v1.name_scope('EhE'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils.tf_fftshift(tf.signal.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask
            image_space_coil_imgs = tf_utils.tf_ifftshift(tf.signal.ifft2d(tf_utils.tf_fftshift(masked_kspace))) * self.scalar
            image_space_comb = tf.reduce_sum(input_tensor=image_space_coil_imgs * tf.math.conj(self.sens_maps), axis=0)

            ispace = image_space_comb + mu * img

        return ispace

    def SSDU_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        and selects only loss mask locations(\Lambda) for computing loss
        """

        with tf.compat.v1.name_scope('SSDU_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils.tf_fftshift(tf.signal.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar
            masked_kspace = kspace * self.mask

        return masked_kspace

    def Supervised_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.compat.v1.name_scope('Supervised_kspace'):
            coil_imgs = self.sens_maps * img
            kspace = tf_utils.tf_fftshift(tf.signal.fft2d(tf_utils.tf_ifftshift(coil_imgs))) / self.scalar

        return kspace
    
    def Supervised_single_kspace(self, img):
        """
        Transforms unrolled network output to k-space
        """

        with tf.compat.v1.name_scope('Supervised_kspace'):
            kspace = tf.squeeze(tf_utils.tf_fftshift(tf.signal.fft2d(tf_utils.tf_ifftshift(tf.expand_dims(img, axis=0)))), axis=0) / self.scalar

        return kspace
    
    def Supervised_single_image(self, kspace):
        """
        Transforms unrolled network output to image
        """

        with tf.compat.v1.name_scope('Supervised_image'):
            image = tf.squeeze(tf_utils.tf_ifftshift(tf.signal.ifft2d(tf_utils.tf_fftshift(tf.expand_dims(kspace, axis=0)))), axis=0) * self.scalar

        return image


def conj_grad(input_elems, mu_param):
    """
    Parameters
    ----------
    input_data : contains tuple of  reg output rhs = E^h*y + mu*z , sens_maps and mask
    rhs = nrow x ncol x 2
    sens_maps : coil sensitivity maps ncoil x nrow x ncol
    mask : nrow x ncol
    mu : penalty parameter

    Encoder : Object instance for performing encoding matrix operations

    Returns
    -------
    data consistency output, nrow x ncol x 2

    """

    rhs, sens_maps, mask = input_elems
    mu_param = tf.complex(mu_param, 0.)
    rhs = tf_utils.tf_real2complex(rhs)

    Encoder = data_consistency(sens_maps, mask)
    cond = lambda i, *_: tf.less(i, args.CG_Iter)

    def body(i, rsold, x, r, p, mu):
        with tf.compat.v1.name_scope('CGIters'):
            Ap = Encoder.EhE_Op(p, mu)
            alpha = tf.complex(rsold / tf.cast(tf.reduce_sum(input_tensor=tf.math.conj(p) * Ap), dtype=tf.float32), 0.)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = tf.cast(tf.reduce_sum(input_tensor=tf.math.conj(r) * r), dtype=tf.float32)
            beta = rsnew / rsold
            beta = tf.complex(beta, 0.)
            p = r + beta * p

        return i + 1, rsnew, x, r, p, mu

    x = tf.zeros_like(rhs)
    i, r, p = 0, rhs, rhs
    rsold = tf.cast(tf.reduce_sum(input_tensor=tf.math.conj(r) * r), dtype=tf.float32, )
    loop_vars = i, rsold, x, r, p, mu_param
    cg_out = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars, name='CGloop', parallel_iterations=1)[2]

    return tf_utils.tf_complex2real(cg_out)


def dc_block(rhs, sens_maps, mask, mu):
    """
    DC block employs conjugate gradient for data consistency,
    """

    def cg_map_func(input_elems):
        cg_output = conj_grad(input_elems, mu)

        return cg_output

    dc_block_output = tf.map_fn(cg_map_func, (rhs, sens_maps, mask), dtype=tf.float32, name='mapCG')

    return dc_block_output


def SSDU_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space at only unseen locations in training (\Lambda locations)
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def ssdu_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.SSDU_kspace(nw_output_enc)

        return nw_output_kspace

    masked_kspace = tf.map_fn(ssdu_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='ssdumapFn')

    return tf_utils.tf_complex2real(masked_kspace)


def Supervised_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.Supervised_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmapFn')

    return tf_utils.tf_complex2real(kspace)


def Supervised_single_kspace_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to k-space
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_kspace = Encoder.Supervised_single_kspace(nw_output_enc)

        return nw_output_kspace

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmap_single_k_Fn')

    return tf_utils.tf_complex2real(kspace)


def Supervised_single_image_transform(nw_output, sens_maps, mask):
    """
    This function transforms unrolled network output to image
    """

    nw_output = tf_utils.tf_real2complex(nw_output)

    def supervised_map_fn(input_elems):
        nw_output_enc, sens_maps_enc, mask_enc = input_elems
        Encoder = data_consistency(sens_maps_enc, mask_enc)
        nw_output_image = Encoder.Supervised_single_image(nw_output_enc)

        return nw_output_image

    kspace = tf.map_fn(supervised_map_fn, (nw_output, sens_maps, mask), dtype=tf.complex64, name='supervisedmap_single_i_Fn')

    return tf_utils.tf_complex2real(kspace)

