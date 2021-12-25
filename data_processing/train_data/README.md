# Prepare Training Data
### Download raw papers from S2ORC 
First, you must download the entire S2ORC corpus to disk, which occupies ~100GB on disk.
Download the full text as jsonl.gz files into a directory called `s2orc_full_texts/`, and download the metadata files as unzipped jsonl files into a directory called `s2orc_metadata`.

```
# Set variables
S2ORC_FULL_TEXTS_PATH=S2ORC_PATH/s2orc_full_texts/
S2ORC_METADATA_PATH=S2ORC_PATH/s2orc_metadata/
```

### Generate auto-tagged dataset labels for papers in S2ORC
```
PWC_S2ORC_MAPPING=s2orc_caches/pwc_dataset_to_s2orc_mapping.pkl
# This command takes several hours to run.
python match_pwc_to_s2orc.py \
    --caches-directory s2orc_caches/ \
    --s2orc-metadata-directory S2ORC_PATH/s2orc_metadata/ \
    --pwc-datasets-file data/datasets.json \
    --pwc-to-s2orc-mapping-file pwc_dataset_to_s2orc_mapping.pkl

python data_processing/train_data/auto_tag_datasets.py \
    --s2orc-full-text-directory $S2ORC_FULL_TEXTS_PATH \
    --s2orc-metadata-directory $S2ORC_METADATA_PATH \
    --pwc-to-s2orc-mapping $PWC_S2ORC_MAPPING \
    --pwc-datasets-file data/datasets.json \
    --output-file tagged_dataset_positives.jsonl

python data_processing/train_data/auto_tag_datasets.py \
    --s2orc-full-text-directory $S2ORC_FULL_TEXTS_PATH \
    --s2orc-metadata-directory $S2ORC_METADATA_PATH \
    --pwc-to-s2orc-mapping $PWC_S2ORC_MAPPING \
    --pwc-datasets-file data/datasets.json \
    --tag-negatives \
    --output-file tagged_dataset_negatives.jsonl
```

### Create anserini index, to be used for hard negative mining via BM25
This part requires the construction of our search corpus (`data/dataset_search_collection.jsonl`). See the main README at the root of this repository for instructions.

```
mkdir -p anserini_search_collections/dataset_search_collection_description_title_only
python data_processing/build_search_corpus/configure_search_collection.py \
    --exclude-abstract \
    --exclude-full-text \
    --full-documents data/dataset_search_collection.jsonl \
    --filtered-documents anserini_search_collections/dataset_search_collection_description_title_only/documents.jsonl

mkdir -p indexes
python -m pyserini.index -collection JsonCollection \
                         -generator DefaultLuceneDocumentGenerator \
                         -threads 1 \
                         -input anserini_search_collections/dataset_search_collection_description_title_only  \
                         -index indexes/dataset_search_collection_description_title_only \
                         -storePositions -storeDocvectors -storeRaw
```

### Scrub dataset mentions from TLDRs and combine into final training dataset
```
python utils/scrub_dataset_mentions_from_tldrs.py \
    train_tldrs.hypo train_tldrs_scrubbed.hypo \
    --datasets-file datasets.json

python data_processing/train_data/merge_tagged_datasets.py \
    --combined-file data/train_data.jsonl \
    --dataset-tldrs train_tldrs_scrubbed.hypo \
    --tagged-positives-file data/tagged_dataset_positives.jsonl \
    --tagged-negatives-file data/tagged_dataset_negatives.jsonl \
    --negative-mining hard \
    --anserini-index indexes/dataset_search_collection_description_title_only \
    --num-negatives 7
```

That's it! After this long process, we now have an auto-tagged set of relevant datasets, ready to use for training a neural retriever, at `data/train_data.jsonl`.