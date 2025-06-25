import tensorflow as tf
import data_consistency as ssdu_dc
import tf_utils
import models.networks as networks
import parser_ops

parser = parser_ops.get_parser()
args = parser.parse_args()


class UnrolledNet():
    """

    Parameters
    ----------
    input_x: batch_size x nrow x ncol x 2
    sens_maps: batch_size x ncoil x nrow x ncol

    trn_mask: batch_size x nrow x ncol, used in data consistency units
    loss_mask: batch_size x nrow x ncol, used to define loss in k-space

    args.nb_unroll_blocks: number of unrolled blocks
    args.nb_res_blocks: number of residual blocks in ResNet

    Returns
    ----------

    x: nw output image
    nw_kspace_output: k-space corresponding nw output at loss mask locations

    x0 : dc output without any regularization.
    all_intermediate_results: all intermediate outputs of regularizer and dc units
    mu: learned penalty parameter


    """

    def __init__(self, input_x, sens_maps, trn_mask, loss_mask):
        self.input_x = input_x
        self.sens_maps = sens_maps
        self.trn_mask = trn_mask
        self.loss_mask = loss_mask
        self.model = self.Unrolled_SSDU()

    def Unrolled_SSDU(self):
        x, denoiser_output, dc_output = self.input_x, self.input_x, self.input_x
        all_intermediate_results = [[0 for _ in range(2)] for _ in range(args.nb_unroll_blocks)]

        mu_init = tf.constant(0., dtype=tf.float32)
        for ee in range(args.necho_GLOB):
            if ee == 0:
                x0 = ssdu_dc.dc_block(self.input_x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu_init)
            else:
                x0 = tf.concat([x0, \
                        ssdu_dc.dc_block(self.input_x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu_init)], axis=-1)
        x = x0

        sens_mask = tf.greater(tf.abs(self.sens_maps[:,0,:,:]), 0)
        sens_mask = tf.where(sens_mask, 1, 0)

        with tf.compat.v1.name_scope('SSDUModel'):
            with tf.compat.v1.variable_scope('Weights', reuse=tf.compat.v1.AUTO_REUSE):
                for i in range(args.nb_unroll_blocks):

                    # mod 020724
                    # for ee in range(args.necho_GLOB):
                    #     x_k_ = tf_utils.tf_real2complex(ssdu_dc.Supervised_single_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask))
                    #     x_k_a_ = tf.angle(x_k_[..., int(args.nrow_GLOB/2)-1, int(args.ncol_GLOB/2)-1])
                    #     x_k_ = tf_utils.tf_complex2real(tf.complex(tf.abs(x_k_), 0.0) * tf.exp(tf.complex(0.0, 1.0) * tf.complex(tf.angle(x_k_) - x_k_a_, 0.0)))
                    #     x_i_ = ssdu_dc.Supervised_single_image_transform(x_k_, self.sens_maps, self.loss_mask)
                    #     if ee == 0:
                    #         x_i = x_i_
                    #         x_k_a = x_k_a_
                    #     else:
                    #         x_i = tf.concat([x_i, x_i_], axis=-1) # 32 224 52 2*e
                    #         x_k_a = tf.concat([x_k_a, x_k_a_], axis=-1)
                    # x = x_i
                    # mod 020724

                    # for ee in range(args.necho_GLOB):
                    #     x_ = tf_utils.tf_real2complex(x[..., ee * 2:(ee + 1) * 2])
                    #     mean_, variance_ = tf.nn.moments(tf.abs(x_), axes=[0, 1, 2])
                    #     mean_, variance_ = tf.expand_dims(mean_, -1), tf.expand_dims(variance_, -1)
                    #     x_norm_ = (x[..., ee * 2:(ee + 1) * 2] - mean_) / tf.sqrt(variance_)
                    #     if ee == 0:
                    #         x_norm = x_norm_
                    #         mean, variance = mean_, variance_
                    #     else:
                    #         x_norm = tf.concat([x_norm, x_norm_], axis=-1)
                    #         mean, variance = tf.concat([mean, mean_], axis=-1), tf.concat([variance, variance_], axis=-1)
                    # x = x_norm

                    x = networks.ResNet(x, args.nb_res_blocks, args.necho_GLOB)
                    denoiser_output = x

                    # for ee in range(args.necho_GLOB):
                    #     x_denorm_ = x[..., ee * 2:(ee + 1) * 2] * tf.sqrt(variance[..., ee]) + mean[..., ee]
                    #     if ee == 0:
                    #         x_denorm = x_denorm_
                    #     else:
                    #         x_denorm = tf.concat([x_denorm, x_denorm_], axis=-1)
                    # x = x_denorm

                    # mod 020724
                    # for kk in range(args.necho_GLOB):
                    #     x_k_ = tf_utils.tf_real2complex(ssdu_dc.Supervised_single_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask))
                    #     x_k_ = tf_utils.tf_complex2real(tf.complex(tf.abs(x_k_), 0.0) * tf.exp(tf.complex(0.0, 1.0) * tf.complex(tf.angle(x_k_) + x_k_a[..., ee], 0.0)))
                    #     x_i_ = ssdu_dc.Supervised_single_image_transform(x_k_, self.sens_maps, self.loss_mask)
                    #     if kk == 0:
                    #         x_i = x_i_
                    #     else:
                    #         x_i = tf.concat([x_i, x_i_], axis=-1) # 32 224 52 2*e
                    # x = x_i
                    # mod 020724

                    mu = networks.mu_param()
                    rhs = self.input_x + mu * x

                    for ee in range(args.necho_GLOB):
                        if ee == 0:
                            x = ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)
                        else:
                            x = tf.concat([x, \
                                    ssdu_dc.dc_block(rhs[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.trn_mask[..., ee], mu)], axis=-1)
                    x = x * tf.cast(tf.repeat(tf.expand_dims(sens_mask, -1), args.necho_GLOB*2, axis=3), tf.float32)
                    dc_output = x

                    # ...................................................................................................
                    all_intermediate_results[i][0] = tf_utils.tf_real2complex(tf.squeeze(denoiser_output))
                    all_intermediate_results[i][1] = tf_utils.tf_real2complex(tf.squeeze(dc_output))

            for ee in range(args.necho_GLOB):
                if ee == 0:
                    nw_kspace_output = ssdu_dc.SSDU_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask[..., ee])
                else:
                    nw_kspace_output = tf.concat([nw_kspace_output, \
                                            ssdu_dc.SSDU_kspace_transform(x[..., ee * 2:(ee + 1) * 2], self.sens_maps, self.loss_mask[..., ee])], axis=-1)
        
        return x, nw_kspace_output, x0, all_intermediate_results, mu
