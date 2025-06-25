import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='ZS-SSL: Zero-Shot Self-Supervised Learning')

    # hyperparameters for the network
    parser.add_argument('--data_opt', type=str, default='_tbme_tutorial',
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default='data/',
                    help='data directory')
    parser.add_argument('--nrow_GLOB', type=int, default=320,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=320,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=20,
                        help='number of coils of the slices in the dataset')
    parser.add_argument('--netl_GLOB', type=int, default=119,
                        help='number of echo train length of the slices in the dataset')
    parser.add_argument('--necho_GLOB', type=int, default=1,
                        help='number of echo or contrast of the slices in the dataset')
    parser.add_argument('--nbasis_GLOB', type=int, default=5,
                        help='number of basis of the slices in the dataset')
    parser.add_argument('--kspace_sum_over_etl', type=int, default=0,
                        help='summing kspace data over etl (i.e. complement sampling)')
    parser.add_argument('--mask_gen_parallel_computation', type=int, default=1,
                        help='mask generation using python parallel computation')
    parser.add_argument('--mask_gen_in_each_iter', type=int, default=1,
                        help='mask generation in each iteration')
    parser.add_argument('--acc_rate', type=int, default=4,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=5,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=10,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=5,
                        help='number of Conjugate Gradient iterations for DC')

    # hyperparameters for the zs-ssl
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask (\Gamma)')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=100,
                        help='number of repetions for the remainder mask (\Omega \ \Gamma) ')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default='/path/to/saved_model/',
                        help='path to pretrained model')                                           
    parser.add_argument('--stop_training', type=int, default=1500,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')
    
    return parser
