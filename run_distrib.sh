export CUDA_VISIBLE_DEVICES=4,5,6,7
export OMP_NUM_THREADS=4

python -m torch.distributed.launch \
    --nproc_per_node=4 main.py \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --logging_steps 10
