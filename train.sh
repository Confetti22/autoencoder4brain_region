python train_classifier.py  \
                        -gpus 0 \
                        -cfg 'config/classifier.yaml' \
                        -slurm \
                        -slurm_ngpus 1 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c001 \
                        -slurm_partition compute \