gpuno="2"
export CUDA_VISIBLE_DEVICES=${gpuno}
export TOKENIZERS_PARALLELISM=true

task="cmudog"
save_path="${task}_GPU${gpuno}.pkl"

## w2v: DGMN
python3 run_cmudog.py \
  --is_training \
  --seed 1234 \
  --task ${task} \
  --hidden_size 300 \
  --emb_size 300 \
  --model "dgmn" \
  --doc_len 40 \
  --seq_len 40 \
  --max_turn_num 4 \
  --max_doc_num 20 \
  --percent 100 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --epochs 5 \
  --valid_focusing_sample 0 \
  --test_focusing_sample 723180 \
  --valid_every 360000 \
  --test_every 360000 \
  --save_path ${save_path}
