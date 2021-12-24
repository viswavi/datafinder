CURRENT_DIR=$(pwd)
METADATA_FILE=$PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata.pkl
UPDATED_METADATA_FILE=$PICKLES_DIRECTORY/scirex_id_to_s2orc_metadata_with_tldrs.pkl

SCIREX_ABSTRACTS_FILE=scirex_abstracts.temp
python extract_abstracts_from_scirex_papers.py --scirex-to-s2orc-metadata-file $METADATA_FILE --tldr-input-file $SCIREX_ABSTRACTS_FILE

TLDR_DIR=${CURRENT_DIR}/tldr/scitldr/
cp $SCIREX_ABSTRACTS_FILE ${TLDR_DIR}/SciTLDR-Data/SciTLDR-A/ctrl/test.source

cd $TLDR_DIR
TLDR_FILE=${CURRENT_DIR}/scirex_tldrs.hypo
CUDA_VISIBLE_DEVICES=3  python scripts/generate.py models/ SciTLDR-Data/SciTLDR-A/ctrl/ ./ --beam 6 --lenpen 0.4 --batch_size 40 --test_fname $TLDR_FILE

cd $CURRENT_DIR
python add_tldrs_to_scirex_metadata.py --scirex-to-s2orc-metadata-file $METADATA_FILE --tldr-file $TLDR_FILE --new-scirex-to-s2orc-metadata-file $UPDATED_METADATA_FILE