# ClusterEA

Source code for ClusterEA: Scalable Entity Alignment with Stochastic Training and Normalized Mini-batch Similarities

## Requirements

pytorch>=1.10.0

tensorflow-gpu==2.4.1 (required for GCN-Align and RREA)

dgl==0.5.3

xgboost

faiss-gpu

...

A full list of required packages is located in ``src/requirements.txt``

## Datasets 

The IDS benchmark is provided by [OpenEA](https://github.com/nju-websoft/OpenEA)

The DBP1M benchmark is  provided by [LargeEA](https://github.com/ZJU-DAILY/LargeEA)

First download and unzip dataset files, place them to the project root folder:

    unzip OpenEA_dataset_v1.1.zip
    unzip mkdata.zip


The __dataset__ (small for IDS15K, medium for IDS100K, large for DBP1M) and  __lang__ (fr or de) parameter controls which benchmark to use.
For example, in the ``src`` folder, setting dataset to small and lang to fr will run on OpenEA EN_FR_15K_V1 dataset.

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


    usage: main.py [-h] [--scale SCALE] [--ds DS] [--lang LANG] [--k K]
               [--it_round IT_ROUND] [--train_ratio TRAIN_RATIO]
               [--epoch EPOCH] [--model MODEL] [--save_folder SAVE_FOLDER]
               [--result_folder RESULT_FOLDER] [--step STEP]
               [--enhance ENHANCE] [--samplers SAMPLERS] [--local_only]
               [--no_csls] [--skip_if_complete]
               [--max_sinkhorn_sz MAX_SINKHORN_SZ]
               [--gcn_max_iter GCN_MAX_ITER] [--cuda] [--faiss_gpu] [--norm]
    
    optional arguments:
      -h, --help            show this help message and exit
      --scale SCALE         dataset scale
      --ds DS               dataset name
      --lang LANG           dataset language (fr, de)
      --k K                 mini-batch number
      --it_round IT_ROUND
      --train_ratio TRAIN_RATIO
      --epoch EPOCH         number of epochs to train
      --model MODEL
      --save_folder SAVE_FOLDER
      --result_folder RESULT_FOLDER
      --step STEP
      --enhance ENHANCE     mini-batch normalization
      --samplers SAMPLERS   C -> CMCS, S->ISCS(src->trg), T-> ISCS(trg->src)
      --local_only
      --no_csls
      --skip_if_complete
      --max_sinkhorn_sz MAX_SINKHORN_SZ
                            max matrix size to run Sinkhorn iteration, if the
                            matrix size is higher than this value, it will
                            calculate kNN search without normalizing to avoid OOM,
                            default is set for 33000^2 (for RTX3090). could be set
                            to higher value in case there is GPU with larger
                            memory
      --gcn_max_iter GCN_MAX_ITER
                            max iteration of GCN for partition
      --cuda                whether to use cuda or not
      --faiss_gpu           whether to use FAISS GPU
      --norm                whether to normalize embeddings

    

## Citation

Not available for now.