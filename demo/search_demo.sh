echo "Must set the PROJ directory before running this script"
# export PROJ=
export JAVA_HOME=$PROJ/jdk-11
export PATH=$JAVA_HOME/bin:$PATH
alias java=$JAVA_HOME/bin/java

source $PROJ/anaconda3/etc/profile.d/conda.sh 
conda activate scirex
python search_demo_bm25.py --anserini-index indexes/dataset_search_collection_description_title_only
