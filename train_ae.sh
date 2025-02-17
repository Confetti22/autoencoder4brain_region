python train_autoencoder.py  \
                        -gpus 0 \
                        -cfg 'config/autoencoder.yaml' \
                        -slurm \
                        -slurm_ngpus 1 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c003 \
                        -slurm_partition compute \