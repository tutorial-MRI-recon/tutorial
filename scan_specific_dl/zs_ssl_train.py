import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import glob
import math
import h5py as h5
import utils
import tf_utils
import parser_ops
import UnrollNet
from multiprocessing import Pool
from multiprocessing import cpu_count
parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

seed = 42

if args.transfer_learning:
    print('Getting weights from trained model:')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.reset_default_graph()
    loadChkPoint_tl = tf.train.latest_checkpoint(args.TL_path)
    with tf.compat.v1.Session(config=config) as sess:
        new_saver = tf.compat.v1.train.import_meta_graph(args.TL_path + '/model_test.meta')
        new_saver.restore(sess, loadChkPoint_tl)
        trainable_variable_collections = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        pretrained_model_weights = [sess.run(v) for v in trainable_variable_collections]


save_dir ='saved_models'
directory = os.path.join(save_dir, 'ZS_SSL' + args.data_opt + '_Rate'+ str(args.acc_rate)+'_'+ str(args.num_reps)+'reps')
if not os.path.exists(directory):
    os.makedirs(directory)

print('..... Create a test model for the testing \n')
test_graph_generator = tf_utils.test_graph(directory)

start_time = time.time()
print('..... ZS-SSL training \n')
tf.compat.v1.reset_default_graph()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

print('..... Loading data for training \n')
data_list = glob.glob(args.data_dir + "/*.mat")
data_list = sorted(data_list, key=os.path.getmtime)
nslice_GLOB = len(data_list)

data = sio.loadmat(data_list[0])
kspace_tmp, original_mask = data['kspace'], data['mask']

sens_maps = np.empty((nslice_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

'''
# h5 read (v7.3 mat file)
f = h5.File(args.data_dir, 'r')
kspace_train, sens_maps, original_mask = f['kspace'][()], f['sens_maps'][()], f['mask'][()]
kspace_train = np.transpose(np.array(kspace_train['real'], dtype=np.float32)) + 1j*np.transpose(np.array(kspace_train['imag'], dtype=np.float32))
sens_maps = np.transpose(np.array(sens_maps['real'], dtype=np.float32)) + 1j*np.transpose(np.array(sens_maps['imag'], dtype=np.float32))
original_mask = np.transpose(np.array(original_mask, dtype=np.float32))
'''
def load_data_parl(ss):
    tic = time.time()
    data = sio.loadmat(data_list[ss])
    kspace_tmp_parl, sens_maps_parl = data['kspace'], data['sens_maps']
    kspace_max_parl = np.max(np.abs(kspace_tmp_parl[:]))
    toc = time.time() - tic
    print("..... Loading data:", ss + 1, "/", nslice_GLOB, ", elapsed_time = ""{:.2f}".format(toc))
    return kspace_max_parl, sens_maps_parl

num_parallel = min([int(cpu_count()/16), nslice_GLOB])
pool = Pool(num_parallel)
kspace_max_parl, sens_maps_parl = zip(*pool.map(load_data_parl, range(nslice_GLOB)))
pool.close()

for ss in range(nslice_GLOB):
    sens_maps[ss, ...] = sens_maps_parl[ss]
kspace_max = max(kspace_max_parl)
del kspace_max_parl, sens_maps_parl
print('..... k-space max ')
print(kspace_max)

print('..... Data shape ')
print('kspace:', kspace_tmp.shape, ', sensitivity maps:', sens_maps[0, ...].shape, ', mask:', original_mask.shape, '\n')

print('..... Normalize the kspace to 0-1 region \n')
# kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

print('..... Generate validation mask \n')
cv_trn_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.float32)
cv_val_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.complex64)
remainder_mask = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.float32)

data = sio.loadmat(data_list[round(nslice_GLOB/2)])
kspace_tmp = data['kspace'] / kspace_max
for ee in range(args.necho_GLOB):
    cv_trn_mask[..., ee], cv_val_mask[..., ee] = utils.uniform_selection(kspace_tmp[..., ee], original_mask[..., ee], \
                                                                        rho=args.rho_val, small_acs_block=(4, 4), seed=seed)

remainder_mask, cv_val_mask = np.copy(cv_trn_mask), np.copy(np.complex64(cv_val_mask))

print('..... Generate validation data \n')
# ref_kspace_val = np.empty((nslice_GLOB, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
ref_kspace_val = np.empty((1, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
# nw_input_val = np.empty((nslice_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
nw_input_val = np.empty((1, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)

# for ss in range(nslice_GLOB):
for ss in range(1): # validate 1 slice
    tic = time.time()
    ss = round(nslice_GLOB/2) # validate 1 slice
    data = sio.loadmat(data_list[ss])
    kspace_tmp = data['kspace'] / kspace_max
    ref_kspace_val_ = kspace_tmp * np.tile(cv_val_mask[..., np.newaxis, :], (1, 1, args.ncoil_GLOB, 1))[np.newaxis]

    for ee in range(args.necho_GLOB):
        # ref_kspace_val[ss, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_val_[..., ee], (0, 3, 1, 2)))
        ref_kspace_val[0, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_val_[..., ee], (0, 3, 1, 2)))
    for ee in range(args.necho_GLOB):
        # nw_input_val[ss, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(
        #                                                     utils.sense1(kspace_tmp[..., ee] * np.tile(cv_trn_mask[..., ee][..., np.newaxis], (1, 1, args.ncoil_GLOB)), \
        #                                                                     sens_maps[ss, ...])[np.newaxis])
        nw_input_val[0, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(
                                                            utils.sense1(kspace_tmp[..., ee] * np.tile(cv_trn_mask[..., ee][..., np.newaxis], (1, 1, args.ncoil_GLOB)), \
                                                                            sens_maps[ss, ...])[np.newaxis])
    toc = time.time() - tic
    print("..... Generate validation data:", ss + 1, "/", nslice_GLOB, ", elapsed_time = ""{:.2f}".format(toc))

sens_maps_trn = np.transpose(sens_maps, (0, 3, 1, 2))

print('..... Make tf. placeholder & dataset \n')
kspaceP = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, None, args.necho_GLOB * 2), name='refkspace')
sens_mapsP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None, None), name='trn_mask')
loss_maskP = tf.compat.v1.placeholder(tf.complex64, shape=(None, None, None, None), name='loss_mask')
nw_inputP = tf.compat.v1.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), name='nw_input')

train_dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP)).shuffle(buffer_size = 10 * args.batchSize).batch(args.batchSize)
cv_dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP)).shuffle(buffer_size = 10 * args.batchSize).batch(args.batchSize)
iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(train_dataset), tf.compat.v1.data.get_output_shapes(train_dataset))
train_iterator = iterator.make_initializer(train_dataset)
cv_iterator = iterator.make_initializer(cv_dataset)

ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')

print('..... Make training model \n')
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model

scalar = tf.constant(0.5, dtype=tf.float32)

# org
# loss = tf.multiply(scalar, tf.norm(tensor=ref_kspace_tensor - nw_output_kspace) / tf.norm(tensor=ref_kspace_tensor)) + \
#        tf.multiply(scalar, tf.norm(tensor=ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(tensor=ref_kspace_tensor, ord=1))

### mod
for ee in range(args.necho_GLOB):
    loss_ = tf.multiply(scalar, tf.norm(tensor=ref_kspace_tensor[..., ee * 2: (ee + 1) * 2] - nw_output_kspace[..., ee * 2: (ee + 1) * 2]) / tf.norm(tensor=ref_kspace_tensor[..., ee * 2: (ee + 1) * 2])) + \
            tf.multiply(scalar, tf.norm(tensor=ref_kspace_tensor[..., ee * 2: (ee + 1) * 2] - nw_output_kspace[..., ee * 2: (ee + 1) * 2], ord=1) / tf.norm(tensor=ref_kspace_tensor[..., ee * 2: (ee + 1) * 2], ord=1))
    if ee == 0:
        loss = loss_
    else:
        loss = loss + loss_
loss = loss / args.necho_GLOB
###

trn_mask, loss_mask = np.empty((nslice_GLOB * args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.complex64), \
                                np.empty((nslice_GLOB * args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.complex64)

ref_kspace = np.empty((nslice_GLOB * args.num_reps, args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
nw_input = np.empty((nslice_GLOB * args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)

if args.mask_gen_parallel_computation == 0:
    for ss in range(nslice_GLOB * args.num_reps):
        tic = time.time()
        data = sio.loadmat(data_list[math.floor(ss / args.num_reps)])
        kspace_tmp = data['kspace'] / kspace_max

        for ee in range(args.necho_GLOB):
            trn_mask[ss, ..., ee], loss_mask[ss, ..., ee] = utils.uniform_selection(kspace_tmp[..., ee], remainder_mask[..., ee], \
                                                                                    rho=args.rho_train, small_acs_block=(4, 4))

        sub_kspace = kspace_tmp * np.tile(trn_mask[ss, ..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))
        ref_kspace_ = kspace_tmp * np.tile(loss_mask[ss, ..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))

        for ee in range(args.necho_GLOB):
            ref_kspace[ss, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_[np.newaxis, ..., ee], (0, 3, 1, 2)))
        for ee in range(args.necho_GLOB):
            nw_input[ss, ..., ee * 2:(ee + 1) * 2] = utils.complex2real(utils.sense1(sub_kspace[..., ee], sens_maps[math.floor(ss / args.num_reps), ...]))

        toc = time.time() - tic
        print("..... making multi-mask:", ss, "elapsed_time = ""{:.2f}".format(toc))

def make_data_reps(ss):
    tic = time.time()
    data = sio.loadmat(data_list[ss])
    kspace_tmp = data['kspace'] / kspace_max

    trn_mask_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.complex64)
    loss_mask_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB), dtype=np.complex64)
    ref_kspace_reps = np.empty((args.ncoil_GLOB, args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)
    nw_input_reps = np.empty((args.nrow_GLOB, args.ncol_GLOB, args.necho_GLOB * 2), dtype=np.float32)

    for ee in range(args.necho_GLOB):
        trn_mask_reps[..., ee], loss_mask_reps[..., ee] = utils.uniform_selection(kspace_tmp[..., ee], remainder_mask[..., ee], \
                                                                                    rho=args.rho_train, small_acs_block=(4, 4))

    sub_kspace = kspace_tmp * np.tile(trn_mask_reps[..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))
    ref_kspace_ = kspace_tmp * np.tile(loss_mask_reps[..., np.newaxis, :].astype(np.complex64), (1, 1, args.ncoil_GLOB, 1))

    for ee in range(args.necho_GLOB):
        ref_kspace_reps[..., ee * 2:(ee + 1) * 2] = utils.complex2real(np.transpose(ref_kspace_[np.newaxis, ..., ee], (0, 3, 1, 2)))

    for ee in range(args.necho_GLOB):
        nw_input_reps[..., ee * 2:(ee + 1) * 2] = utils.complex2real(utils.sense1(sub_kspace[..., ee], sens_maps[ss, ...]))

    toc = time.time() - tic
    print("..... making multi-mask:", ss, "elapsed_time = ""{:.2f}".format(toc))
    return trn_mask_reps, loss_mask_reps, ref_kspace_reps, nw_input_reps

if args.mask_gen_parallel_computation == 1:
    for rr in range(args.num_reps):
        num_parallel = min([int(cpu_count()/16), nslice_GLOB])
        pool = Pool(num_parallel)
        trn_mask_reps, loss_mask_reps, ref_kspace_reps, nw_input_reps = zip(*pool.map(make_data_reps, range(nslice_GLOB)))
        pool.close()
        for ss in range(nslice_GLOB):
            trn_mask[ss + rr * nslice_GLOB, ...], loss_mask[ss + rr * nslice_GLOB, ...], \
            ref_kspace[ss + rr * nslice_GLOB, ...], nw_input[ss + rr * nslice_GLOB, ...] = \
            trn_mask_reps[ss], loss_mask_reps[ss], ref_kspace_reps[ss], nw_input_reps[ss]
    del trn_mask_reps, loss_mask_reps, ref_kspace_reps, nw_input_reps

all_trainable_vars = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in tf.compat.v1.trainable_variables()])
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss) # org

# mod v11
# Gradient clipping section
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
grads_and_vars = opt.compute_gradients(loss)
grads, vars = zip(*grads_and_vars)
clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=5.0)
clipped_grads_and_vars = list(zip(clipped_grads, vars))
optimizer = opt.apply_gradients(clipped_grads_and_vars)
# mod

saver = tf.compat.v1.train.Saver(max_to_keep=1) # only keep the model corresponding to lowest validation error
sess_trn_filename = os.path.join(directory, 'model')
totalLoss, totalTime = [], []
total_val_loss = []
avg_cost = 0

print('..... Start tf.session \n')
lowest_val_loss = np.inf
with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print('Number of trainable parameters: ', sess.run(all_trainable_vars), '\n')
    if args.mask_gen_in_each_iter == 0:
        feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps_trn}

    print('..... Training \n')
    # if for args.stop_training consecutive epochs validation loss doesnt go below the lowest val loss,\
    # stop the training
    if args.transfer_learning:
        print('transferring weights from pretrained model to the new model:')
        trainable_collection_test = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        initialize_model_weights = [v for v in trainable_collection_test]
        for ii in range(len(initialize_model_weights)):
            sess.run(initialize_model_weights[ii].assign(pretrained_model_weights[ii]))

    np.random.seed(seed)
    ep, val_loss_tracker = 0, 0
    while ep<args.epochs and val_loss_tracker<args.stop_training:
        if args.mask_gen_in_each_iter == 1:
            # v9_mod, change mask for each iteration
            np.random.seed()
            ss = np.random.randint(nslice_GLOB * args.num_reps)
            ref_kspace_batch, nw_input_batch = ref_kspace[ss, ...][np.newaxis], nw_input[ss, ...][np.newaxis]
            trn_mask_batch, loss_mask_batch = trn_mask[ss, ...][np.newaxis], loss_mask[ss, ...][np.newaxis]
            sens_maps_trn_batch = sens_maps_trn[ss % nslice_GLOB, ...][np.newaxis]
            feedDict = {kspaceP: ref_kspace_batch, nw_inputP: nw_input_batch, trn_maskP: trn_mask_batch, loss_maskP: loss_mask_batch, \
                        sens_mapsP: sens_maps_trn_batch}

        sess.run(train_iterator, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:
            if args.mask_gen_in_each_iter == 0:
                for ss in range(nslice_GLOB * args.num_reps):
                    tmp, _, _ = sess.run([loss, update_ops, optimizer])
                    avg_cost += tmp / (nslice_GLOB * args.num_reps)
            else:
                # v9_mod
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp

            toc = time.time() - tic
            totalLoss.append(avg_cost)
        except tf.errors.OutOfRangeError:
            pass
        # perform validation
        sess.run(cv_iterator, feed_dict={kspaceP: ref_kspace_val, nw_inputP: nw_input_val, trn_maskP: cv_trn_mask[np.newaxis], loss_maskP: cv_val_mask[np.newaxis], \
                                            sens_mapsP: sens_maps_trn[round(nslice_GLOB/2)][np.newaxis]})
        val_loss = sess.run([loss])[0]
        total_val_loss.append(val_loss)
        print("Epoch:", ep, "elapsed_time =""{:.2f}".format(toc), "trn loss =", "{:.5f}".format(avg_cost)," val loss =", "{:.5f}".format(val_loss))
        if val_loss<=lowest_val_loss:
            lowest_val_loss = val_loss    
            saver.save(sess, sess_trn_filename, global_step=ep)
            val_loss_tracker = 0 #reset the val loss tracker each time a new lowest val error is achieved
        else:
            val_loss_tracker += 1
        sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': totalLoss, 'val_loss': total_val_loss})
        ep += 1

end_time = time.time()
print('Training completed in  ', str(ep), ' epochs, ',((end_time - start_time) / 60), ' minutes')
