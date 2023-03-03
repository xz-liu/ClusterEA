# ClusterEA

Source code for [ClusterEA: Scalable Entity Alignment with Stochastic Training and Normalized Mini-batch Similarities](https://arxiv.org/abs/2205.10312)

## Installation

    pytorch
    dgl
    xgboost   
    faiss-gpu
    tensorflow-gpu==2.4.1 (not mandatory, for running tensorflow version of GCN-Align and RREA)
    ...

A full list of required packages is located in ``src/requirements.txt``

Notes:

We use the CUDA version of XGBoost, for installation, we recommend users to 
[build it from source](https://xgboost.readthedocs.io/en/stable/build.html).  

The TensorFlow package is not necessary for running ClusterEA. 
The original code of incorporated models are written in TensorFlow,
 we include these code for evaluating the correctness of our re-implementation.
 
If you cannot find the ``nxmetis`` package, please refer to [this page](https://github.com/networkx/networkx-metis).
 
All the code are tested in one RTX3090 GPU with CUDA 11.3.


## Monitoring the GPU Memory Usage


We recommend users to use the [NVIDIA NSight Systems](https://developer.nvidia.com/nsight-systems).
for monitoring the GPU status.

To monitor the GPU memory usage, we need to set the corresponding flag to True.
Note that the profiling system will cause computation overhead. [Click here for details.](https://developer.nvidia.com/blog/understanding-the-visualization-of-overhead-and-latency-in-nsight-systems/)
A sample command is as follows.

    nsys profile --cuda-memory-usage true -o [PROFILE_SAVE_NAME] python -u main.py [ARGUMENTS]


## Dataset 

The IDS benchmark is provided by [OpenEA](https://github.com/nju-websoft/OpenEA). 
We use the 2.0 version of IDS dataset to avoid name bias issue.

The DBP1M benchmark is provided by [LargeEA](https://github.com/ZJU-DAILY/LargeEA).

First download and unzip dataset files, place them to the project root folder:

    unzip OpenEA_dataset_v2.0.zip
    unzip mkdata.zip


The __scale__ (small for IDS15K, medium for IDS100K, large for DBP1M) and  __lang__ (fr or de) parameter controls which benchmark to use.
For example, in the ``src`` folder, setting scale to small and lang to fr will run on OpenEA EN_FR_15K_V1 dataset.

## Run

Take __DBP1M(EN-FR)__ and __Dual-AMN-S__ as an example:

Make sure the folder for results is created:

    cd src/
    mkdir tmp
    mkdir result

### Stochastic Training

First get the embeddings of all entities via Stochastic Training of Dual-AMN.

    python main.py --step 1 --scale large --lang fr --model dual-large
    
### ClusterSampler

The ClusterSampler uses result of Stochastic Training. Make sure run Stochastic Training first.

To run ClusterSampler with the embedding provided by Dual-AMN-S model: 

    python main.py --step 2 --scale large --lang fr --model dual-large


### Similarity Fusion and Eval

Finally, fuse the similarity

    python main.py --step 3 --scale large --lang fr --model dual-large
    

### Other parameter settings

Feel free to play with hyper-parameters of ClusterEA to obtain a better result.

A help for all arguments is provided as follows:


    usage: main.py [-h] [--scale SCALE] [--ds DS] [--lang LANG] [--k K] [--it_round IT_ROUND] [--train_ratio TRAIN_RATIO] [--epoch EPOCH] [--model MODEL] [--save_folder SAVE_FOLDER]
                   [--result_folder RESULT_FOLDER] [--step STEP] [--enhance ENHANCE] [--samplers SAMPLERS] [--local_only] [--no_csls] [--skip_if_complete] [--max_sinkhorn_sz MAX_SINKHORN_SZ]
                   [--gcn_max_iter GCN_MAX_ITER] [--cuda] [--faiss_gpu] [--norm]
    
    optional arguments:
      -h, --help            show this help message and exit
      --scale SCALE         dataset scale, small -> IDS15Kmedium -> IDS100Klarge -> DBP1M
      --ds DS               dataset name
      --lang LANG           dataset language (fr, de)
      --k K                 mini-batch number
      --it_round IT_ROUND
      --train_ratio TRAIN_RATIO
      --epoch EPOCH         number of epochs to train
      --model MODEL         model used for training, including [gcn-align, rrea, dual-amn, gcn-large, rrea-large, dual-large].'-large' indicates the sampling version of the model
      --save_folder SAVE_FOLDER
      --result_folder RESULT_FOLDER
      --step STEP
      --enhance ENHANCE     mini-batch normalization
      --samplers SAMPLERS   C -> CMCS, S->ISCS(src->trg), T-> ISCS(trg->src)
      --local_only
      --no_csls
      --skip_if_complete
      --max_sinkhorn_sz MAX_SINKHORN_SZ
                            max matrix size to run Sinkhorn iteration, if the matrix size is higher than this value, it will calculate kNN search without normalizing to avoid OOM, default is set for 33000^2
                            (for RTX3090). could be set to higher value in case there is GPU with larger memory
      --gcn_max_iter GCN_MAX_ITER
                            max iteration of GCN for partition
      --cuda                whether to use cuda or not
      --faiss_gpu           whether to use FAISS GPU
      --norm                whether to normalize embeddings


## Citation


    @inproceedings{ClusterEA,
      author    = {Yunjun Gao and Xiaoze Liu and Junyang Wu and Tianyi Li and Pengfei Wang and Lu Chen},
      title     = {ClusterEA: Scalable Entity Alignment with Stochastic Training
    and Normalized Mini-batch Similarities},
      booktitle = {KDD},
      year = {2022}
    }
