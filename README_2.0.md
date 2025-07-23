Currently working by running:


torchrun --standalone --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vitl14_rxrx3_core.yaml --output-dir .checkpoints train.dataset_path=RXRX3_CORE:split=TRAIN:root=.:extra="" 

however failes due to _handles error
