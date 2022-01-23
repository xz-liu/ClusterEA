import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # My arguments
    parser.add_argument('--scale', type=str, default='small', help='dataset scale, '
                                                                   'small -> IDS15K'
                                                                   'medium -> IDS100K'
                                                                   'large -> DBP1M')
    parser.add_argument('--ds', type=str, default='ids', help='dataset name')
    parser.add_argument('--lang', type=str, default='fr', help='dataset language (fr, de)')
    parser.add_argument('--k', type=int, default=-1, help='mini-batch number')
    parser.add_argument('--it_round', type=int, default=1)
    parser.add_argument('--train_ratio', type=int, default=30)
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
    parser.add_argument('--model', type=str, default='dual-large', help='model used for training, '
                                                                        'including [gcn-align, rrea, dual-amn,'
                                                                        ' gcn-large, rrea-large, dual-large].'
                                                                        '\'-large\' indicates the '
                                                                        'sampling version of the model')
    parser.add_argument("--save_folder", type=str, default='tmp/')
    parser.add_argument("--result_folder", type=str, default='result/')
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enhance", type=str, default='sinkhorn', help='mini-batch normalization')
    parser.add_argument("--samplers", type=str, default='CST', help='C -> CMCS, S->ISCS(src->trg), T-> ISCS(trg->src)')
    parser.add_argument('--local_only', action='store_true', default=False)
    parser.add_argument('--no_csls', action='store_true', default=False)
    parser.add_argument("--skip_if_complete", action='store_true', default=False)
    parser.add_argument("--max_sinkhorn_sz", type=int, default=33000,
                        help="max matrix size to run Sinkhorn iteration"
                             ", if the matrix size is higher than this value"
                             ", it will calculate kNN search without normalizing to avoid OOM"
                             ", default is set for 33000^2 (for RTX3090)."
                             " could be set to higher value in case there is GPU with larger memory")
    parser.add_argument("--gcn_max_iter", type=int, default=-1, help="max iteration of GCN for partition")
    parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
    parser.add_argument("--faiss_gpu", action="store_true", default=True, help="whether to use FAISS GPU")
    parser.add_argument("--norm", action="store_true", default=True, help="whether to normalize embeddings")
    return parser.parse_args()


global_arguments = get_arguments()

from framework import *
import time
import utils

scale = global_arguments.scale
ds = global_arguments.ds
lang = global_arguments.lang
train_ratio = global_arguments.train_ratio

device = 'cuda' if global_arguments.cuda else 'cpu'
norm = global_arguments.norm
n_semi_iter = global_arguments.it_round
partition_k = global_arguments.k
model_name = global_arguments.model
faiss_use_gpu = global_arguments.faiss_gpu
max_sinkhorn_sz = global_arguments.max_sinkhorn_sz
gcn_max_iter = global_arguments.gcn_max_iter
save_folder = global_arguments.save_folder
enhance = global_arguments.enhance
sampler_methods = global_arguments.samplers
fuse_global = not global_arguments.local_only
apply_csls = not global_arguments.no_csls
result_folder = global_arguments.result_folder
skip_if_complete = global_arguments.skip_if_complete
model_dims = {'gcn-align': 200,
              'rrea': 600,
              'rrea-large': 600,
              'dual-amn': 768,
              'dual-large': 768,
              'gcn-large': 200}

PHASE_TRAINING = 1
PHASE_PARTITION = 2

if partition_k == -1:
    partition_k = dict(small=5, medium=10, large=30)[scale]

if gcn_max_iter == -1:
    gcn_max_iter = dict(small=800, medium=1500, large=3000)[scale]
if global_arguments.epoch == -1:
    train_epoch = \
        {'gcn-align': [2000] * n_semi_iter, 'rrea': [1200] * n_semi_iter, 'dual-amn': [20] + [5] * (n_semi_iter - 1),
         'gcn-large': [50], 'dual-large': [20], 'rrea-large': [50]}[
            model_name]
    if model_name in ['dual-large'] and scale == 'large':
        train_epoch = [10]
else:
    train_epoch = global_arguments.epoch


def ablation_args(val, default_val, name=''):
    if val != default_val:
        return f'_{name}{val}'
    return ''


def get_suffix(phase):
    now = 'embeddings.pkl' if PHASE_TRAINING == phase else 'sim.pkl'
    if phase == PHASE_PARTITION:
        now += f"_{scale}_{ds}_{lang}_{model_name}_gcn{gcn_max_iter}_k{partition_k}_it{n_semi_iter}_norm{norm}"

        now += ablation_args(enhance, 'sinkhorn')
        # now += ablation_args(sampler_methods, 'CST')
    elif phase == PHASE_TRAINING:
        now += f"_{scale}_{ds}_{lang}_{model_name}_it{n_semi_iter}"
    else:
        raise NotImplementedError
    now += ablation_args(train_ratio, 30)
    return now


def save_curr_objs(objs, phase):
    saveobj(objs, save_folder + get_suffix(phase))


def load_curr_objs(phase):
    try:
        return readobj(save_folder + get_suffix(phase))
    except:
        return readobj(save_folder + get_suffix(phase))


if model_name == 'rrea':
    norm = True


def step1_training():
    if skip_if_complete:
        ok = False
        try:
            load_curr_objs(PHASE_TRAINING)
        except:
            ok = True

        if ok:
            add_log('skip_training', True)
            return

    ea = load_dataset(scale, ds, lang, train_ratio=train_ratio * 0.01)
    from align_batch import get_whole_batch
    # TODO fix training
    whole_batch = get_whole_batch(ea, backbone=model_name)
    model = whole_batch.model
    tic = time.time()
    for ep in range(n_semi_iter):
        model.train1step(train_epoch[ep])
        if ep < n_semi_iter - 1:
            # embeddings = model.get_curr_embeddings('cpu')
            model.mraea_iteration()
    # model.update_trainset(get_seed(ea, embeddings))
    embeddings = model.get_curr_embeddings()
    toc = time.time()
    add_log('training time', toc - tic)
    save_curr_objs((ea, embeddings), PHASE_TRAINING)

    del ea, embeddings


def nn_csls_evaluate_of_embeddings():
    ea, embeddings = load_curr_objs(PHASE_TRAINING)

    framework = LargeFramework(ea, max_sinkhorn_sz, device)
    time0 = time.time()
    nn = framework.create_batch_sim(embeddings[0], embeddings[1], enhance='none', whole_batch=True)
    csls = framework.csls_matrix(nn, *embeddings, gpu=faiss_use_gpu, )
    time1 = time.time()
    _, nn_acc = framework.eval_sim(nn)
    _, csls_acc = framework.eval_sim(csls)
    del csls
    add_log('whole_batch_knn_acc', nn_acc)
    add_log('whole_batch_csls_acc', csls_acc)
    add_log('csls_time', time1 - time0)


# 3731002
def step2_batch_similarity():
    if skip_if_complete:
        ok = True
        try:
            load_curr_objs(PHASE_PARTITION)
        except:
            ok = False

        if ok:
            add_log('skip_partition', True)
            return
    ea, embeddings = load_curr_objs(PHASE_TRAINING)
    framework = LargeFramework(ea, max_sinkhorn_sz, device)
    add_log('important_args', (partition_k, enhance, sampler_methods))
    if 'C' in sampler_methods:
        kmeans_xgb_partition_matrix = \
            framework.get_partition_similarity_matrix(partition_k, enhance=enhance, embeddings=embeddings,
                                                      partition_method='kmeans', model='xgb', norm=norm)
    else:
        kmeans_xgb_partition_matrix = torch.tensor(0)
    if 'S' in sampler_methods:
        metis_gcn_partition_matrix = \
            framework.get_partition_similarity_matrix(partition_k, enhance=enhance, embeddings=embeddings,
                                                      partition_method='metis', model='gcn',
                                                      model_dim=model_dims[model_name], max_iter=gcn_max_iter,
                                                      norm=norm,
                                                      src=0)
    else:
        metis_gcn_partition_matrix = torch.tensor(0)

    if 'T' in sampler_methods:
        metis_gcn_partition_matrix_t = \
            framework.get_partition_similarity_matrix(partition_k, enhance=enhance, embeddings=embeddings,
                                                      partition_method='metis', model='gcn',
                                                      model_dim=model_dims[model_name], max_iter=gcn_max_iter,
                                                      norm=norm,
                                                      src=1)
    else:
        metis_gcn_partition_matrix_t = torch.tensor(0)
    # framework.eval_sim(metis_gcn_partition_matrix)
    # framework.eval_sim(metis_gcn_partition_matrix_t)
    # metis_gcn_partition_matrix = metis_gcn_partition_matrix

    # framework.eval_sim(metis_gcn_partition_matrix)
    # framework.get_metis_cps_similarity_matrix(k=partition_k)
    save_curr_objs((framework, metis_gcn_partition_matrix, metis_gcn_partition_matrix_t, kmeans_xgb_partition_matrix),
                   PHASE_PARTITION)


def understandable_sampler_args():
    cst = dict(C='KMeansXGB', S='MetisGCN_s2t', T='MetisGCN_t2s')
    return list(map(lambda x: cst[x], sampler_methods))


@torch.no_grad()
def step3_fuse_and_csls():
    add_log('important_args', (partition_k,
                               enhance,
                               understandable_sampler_args(),
                               f'Global_{fuse_global}',
                               f'CSLS_{apply_csls}'))
    framework, metis_gcn_partition_matrix, metis_gcn_partition_matrix_t, kmeans_xgb_partition_matrix = \
        load_curr_objs(PHASE_PARTITION)
    ea, embeddings = load_curr_objs(PHASE_TRAINING)
    if 'C' not in sampler_methods:
        kmeans_xgb_partition_matrix = None
    if 'S' not in sampler_methods:
        metis_gcn_partition_matrix = None
    if 'T' not in sampler_methods:
        metis_gcn_partition_matrix_t = None

    if fuse_global:
        tic = time.time()
        global_matrix = framework.create_batch_sim(*embeddings, norm=norm, src=0, whole_batch=True)
        global_matrix_t = framework.create_batch_sim(*embeddings, norm=norm, src=1, whole_batch=True)
        partial_csls = framework.csls_matrix((
            global_matrix.to(device), global_matrix_t.to(device).t()), gpu=faiss_use_gpu, apply_csls=apply_csls,
            *embeddings)
        # partial_csls = framework.get_test_sim(partial_csls)
        # partial_csls = sparse_minmax(partial_csls.to(device)).cpu()
        csls_time = time.time() - tic
    else:
        partial_csls = None
        csls_time = 0
    # framework.eval_sim(metis_gcn_partition_matrix.to(device) + kmeans_xgb_partition_matrix.to(device))
    # framework.eval_sim(
    #     metis_gcn_partition_matrix.to(device) + kmeans_xgb_partition_matrix.to(device) + global_matrix.to(
    #         device) + global_matrix_t.t().to(device))

    tic = time.time()
    # metis_gcn_partition_matrix = metis_gcn_partition_matrix + metis_gcn_partition_matrix_t
    # framework.eval_sim(
    #     partial_csls.to(device) +
    #     kmeans_xgb_partition_matrix.to(device) +
    #     metis_gcn_partition_matrix.to(device))
    final_csls = framework.csls_matrix((
        metis_gcn_partition_matrix,
        metis_gcn_partition_matrix_t,
        kmeans_xgb_partition_matrix,
        partial_csls), *embeddings, csls_k=10, apply_csls=apply_csls and fuse_global, gpu=faiss_use_gpu)
    fuse_time = time.time() - tic

    final_csls, acc = framework.eval_sim(final_csls)
    return acc, csls_time, fuse_time


def evaluate_partition_methods():
    print('K is', partition_k)
    add_log('current_partition_k', partition_k)
    ea, embeddings = load_curr_objs(PHASE_TRAINING)
    framework = LargeFramework(ea, max_sinkhorn_sz, device)
    embeddings = list(map(norm_process, embeddings))
    batch_overlaps = {}
    partition_time = {}

    # ------------------------
    partition = Partition(ea)
    tm = time.time()
    partition, src_nodes, trg_nodes, src_train, trg_train = \
        framework.get_partition_nodes(partition_k, embeddings=embeddings,
                                      partition_method='kmeans', model='xgb', partition=partition)
    partition_time['kmeans'] = time.time() - tm
    batch_overlaps['kmeans'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)

    # ------------------------
    # partition = Partition(ea)
    # tm = time.time()
    # src_nodes, trg_nodes, src_train, trg_train = \
    #     partition.random_partition(0, partition_k, partition_k)
    # partition_time['vps'] = time.time() - tm
    # batch_overlaps['vps'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)

    # ------------------------

    # partition = Partition(ea)
    # tm = time.time()
    # src_nodes, trg_nodes, src_train, trg_train = \
    #     partition.partition(0, partition_k, partition_k)
    # partition_time['metis_cps_src=0'] = time.time() - tm
    # batch_overlaps['metis_cps_src=0'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)
    #
    # # ------------------------
    #
    partition = Partition(ea)
    tm = time.time()
    trg_nodes, src_nodes, trg_train, src_train = \
        partition.partition(1, partition_k, partition_k)
    partition_time['metis_cps_src=1'] = time.time() - tm
    batch_overlaps['metis_cps_src=1'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)

    # ------------------------
    partition = Partition(ea)
    tm = time.time()
    partition, src_nodes, trg_nodes, src_train, trg_train = \
        framework.get_partition_nodes(partition_k, embeddings=embeddings,
                                      partition_method='metis', model='gcn',
                                      model_dim=model_dims[model_name], max_iter=gcn_max_iter,
                                      src=0, partition=partition)
    partition_time['metis_gcn_src=0'] = time.time() - tm
    batch_overlaps['metis_gcn_src=0'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)

    # ------------------------
    partition = Partition(ea)
    tm = time.time()
    partition, src_nodes, trg_nodes, src_train, trg_train = \
        framework.get_partition_nodes(partition_k, embeddings=embeddings,
                                      partition_method='metis', model='gcn', max_iter=gcn_max_iter,
                                      model_dim=model_dims[model_name],
                                      src=1, partition=partition)
    partition_time['metis_gcn_src=1'] = time.time() - tm
    batch_overlaps['metis_gcn_src=1'] = partition.eval_partition(src_nodes, trg_nodes, src_train, trg_train)
    for k, v in batch_overlaps.items():
        print(f'{k}, time={partition_time[k]}')
        print(f'\t{v}')
    return partition_time, batch_overlaps


def run():
    log_file = f'{result_folder}{scale}_{lang}_{model_name}'
    torch.cuda.set_device(0)
    # eval_large()
    step = global_arguments.step
    if step == 1:
        print('------------Training------------------')
        step1_training()
    elif step == 2:
        print('------------Get Sim Matrix------------------')
        step2_batch_similarity()
    elif step == 3:
        print('------------Fuse Sim Matrix------------------')
        acc, t1, t2 = step3_fuse_and_csls()
        add_log('acc', acc)
        add_log('global_time', t1)
        add_log('fuse_time', t2)

    elif step == 4:
        print('------------Evaluate Partition Methods------------------')
        part_time, overlap = evaluate_partition_methods()
        for k, v in overlap.items():
            add_log(k, f'time = {part_time[k]}\t {v}')
    elif step == 5:
        print('------------Evaluate Embeddings------------------')
        nn_csls_evaluate_of_embeddings()
    with open(log_file, 'a') as f:
        f.write('---------------------\n')
        f.write(f'step : {step}\n')
        for k, v in global_dict.items():
            f.write(f'{k} : {v}\n')


#
# def step3_fuse_and_evaluation():
#     framework, metis_gcn_partition_matrix, kmeans_xgb_partition_matrix, global_matrix, global_matrix_t = \
#         load_curr_objs('sim.pkl', PHASE_PARTITION)
#     ea, embeddings = load_curr_objs('embeddings.pkl', PHASE_TRAINING)
#     # framework.to(device)
#     framework.eval_sim(kmeans_xgb_partition_matrix)
#     framework.eval_sim(metis_gcn_partition_matrix)
#     framework.eval_sim(kmeans_xgb_partition_matrix + metis_gcn_partition_matrix)
#     framework.eval_sim(
#         metis_gcn_partition_matrix.to(device) +
#         kmeans_xgb_partition_matrix.to(device) +
#         global_matrix.to(device) + global_matrix_t.t().to(device))
#
#     framework.eval_sim(
#         metis_gcn_partition_matrix.to(device) +
#         kmeans_xgb_partition_matrix.to(device) +
#         global_matrix.to(device))
#
#     framework.eval_sim(
#         metis_gcn_partition_matrix.to(device) +
#         kmeans_xgb_partition_matrix.to(device) +
#         (global_matrix.to(device) + global_matrix_t.t().to(device)) / 2)
#
#     partial_csls = csls_matrix((
#         global_matrix.to(device) * 2, global_matrix_t.to(device).t() * 2), gpu=faiss_use_gpu, *embeddings)
#     partial_csls = framework.eval_sim(partial_csls)
#     sparse_minmax(partial_csls.to(device))
#     framework.eval_sim(partial_csls.to(device) * 0.5
#                        + metis_gcn_partition_matrix.to(device)
#                        + kmeans_xgb_partition_matrix.to(device))
#
#     final_csls = csls_matrix((
#         metis_gcn_partition_matrix.to(device),
#         kmeans_xgb_partition_matrix.to(device),
#         global_matrix.to(device) * 2,
#         global_matrix_t.t().to(device) * 2), *embeddings, csls_k=10, gpu=faiss_use_gpu)
#     final_csls = framework.eval_sim(final_csls)
#     final_csls = csls_matrix((
#         sparse_minmax(metis_gcn_partition_matrix.to(device), in_place=False),
#         sparse_minmax(kmeans_xgb_partition_matrix.to(device), in_place=False),
#         sparse_minmax(global_matrix.to(device), in_place=False),
#         sparse_minmax(global_matrix_t.t().to(device))), *embeddings, csls_k=10, gpu=faiss_use_gpu)
#     final_csls = framework.eval_sim(final_csls)

if __name__ == '__main__':
    run()
