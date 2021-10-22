export PROJ=/projects/ogma1/vijayv/
export JAVA_HOME=$PROJ/jdk-11
export PATH=$JAVA_HOME/bin:$PATH
alias java=$JAVA_HOME/bin/java

source /projects/ogma1/vijayv/anaconda3/etc/profile.d/conda.sh 
conda activate scirex
python search_demo_bm25.py --anserini-index indexes/dataset_search_collection_no_abstracts_or_paper_text_jsonl
