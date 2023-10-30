# Training

### Prepare data
```
python retrieval/biencoder/tevatron_scripts/prepare_tevatron_data.py \
    --tagged-datasets-file data/train_data.jsonl \
    --search-collection data/dataset_search_collection.jsonl \
    --test-queries data/test_data.jsonl \
    --output-training-directory tevatron_data/training_data \
    --output-metadata-directory tevatron_data/metadata \
    --output-search-directory tevatron_data/search_data \
    --output-query-file tevatron_data/test_queries.jsonl
```

### Train model
```
mkdir tevatron_models
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.train \
    --output_dir tevatron_models/scibert \
    --model_name_or_path allenai/scibert_scivocab_uncased \
    --do_train \
    --save_steps 20000 \
    --train_dir tevatron_data/training_data \
    --fp16 \
    --per_device_train_batch_size 11 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --dataloader_num_workers 2
```

# Retrieval

### Encode Corpus
```
ENCODE_DIR=tevatron_data/search_encoded
mkdir $ENCODE_DIR
CUDA_VISIBLE_DEVICES=0 python -m tevatron.driver.encode \
    --output_dir=temp \
    --tokenizer_name allenai/scibert_scivocab_uncased \
    --model_name_or_path tevatron_models/scibert \
    --fp16 \
    --per_device_eval_batch_size 128 \
    --encode_in_path tevatron_data/search_data/0.json \
    --encoded_save_path $ENCODE_DIR/0.pt
```

### Encode Query
```
ENCODE_QRY_DIR=tevatron_data/test_queries_encoded
OUTDIR=temp
MODEL_DIR=tevatron_models/scibert
QUERY=tevatron_data/test_queries.jsonl
mkdir $ENCODE_QRY_DIR
CUDA_VISIBLE_DEVICES=2 python -m tevatron.driver.encode \
    --fp16 \
    --output_dir=$OUTDIR \
    --tokenizer_name allenai/scibert_scivocab_uncased \
    --model_name_or_path $MODEL_DIR \
    --per_device_eval_batch_size 156 \
    --encode_in_path $QUERY \
    --encoded_save_path $ENCODE_QRY_DIR/query.pt
```

### Retrieve Documents for Queries
```
ENCODE_QRY_DIR=tevatron_data/test_queries_encoded
ENCODE_DIR=tevatron_data/search_encoded
DEPTH=100
CUDA_VISIBLE_DEVICES=2 python -m tevatron.faiss_retriever \
    --query_reps $ENCODE_QRY_DIR/query.pt \
    --passage_reps $ENCODE_DIR/'*.pt' \
    --depth $DEPTH \
    --batch_size -1 \
    --save_text \
    --save_ranking_to tevatron_models/scibert/rank.tsv
```

### Convert tevatron output to TREC format
For natural language queries:
```
python retrieval/biencoder/tevatron_scripts/convert_tevatron_output_to_trec_eval.py \
    --output-trec-file tevatron_data/tevatron.trec \
    --tevatron-ranking tevatron_models/scibert/rank.tsv \
    --id2dataset tevatron_data/metadata/id2dataset.json \
    --test-queries tevatron_data/test_queries.jsonl \
    --search-collection data/dataset_search_collection.jsonl \
    --depth 5
```

For keyphrase queries:
```
python retrieval/biencoder/tevatron_scripts/convert_tevatron_output_to_trec_eval.py \
    --output-trec-file tevatron_data/tevatron.trec \
    --tevatron-ranking tevatron_models/scibert/rank.tsv \
    --id2dataset tevatron_data/metadata/id2dataset.json \
    --test-queries <path to test_queries.jsonl file for NATURAL LANGUAGE queries \
    --search-collection data/dataset_search_collection.jsonl \
    --depth 5
```
(the command here is different than for natural language queries, because we want to map the keyphrase query IDs to the natural language query IDs, since our ground truth is already processed to include natural language query IDs).
