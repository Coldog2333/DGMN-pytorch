gpuno="0,1,2,3"
n_gpu=4
export TOKENIZERS_PARALLELISM=true
export NCCL_P2P_LEVEL=PXB #      uncomment if you use A100.

# w2v: DGMN
CUDA_VISIBLE_DEVICES=${gpuno} python3 -m torch.distributed.launch --nproc_per_node=${n_gpu} --master_port 23333 run_cmudog.py \
  --seed 1234 \
  --task "cmudog" \
  --fp16 \
  --fp16_opt_level "O1" \
  --hidden_size 256 \
  --emb_size 400 \
  --gamma 0.2 \
  --n_layer 3 \
  --model "dgmn" \
  --doc_len 40 \
  --seq_len 40 \
  --word_len 18 \
  --max_turn_num 4 \
  --max_doc_num 20 \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --epochs 10 \
  --valid_focusing_sample 180795 \
  --test_focusing_sample 361590 \
  --valid_every 25000 \
  --test_every 50000