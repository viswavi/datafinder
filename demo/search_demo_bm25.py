'''
python search_demo.py --anserini-index indexes/dataset_search_collection_no_abstracts_or_paper_text_jsonl
python search_demo.py --remove-punctuation --remove-stopwords --anserini-index indexes/dataset_search_collection_no_abstracts_or_paper_text_jsonl
'''
from nltk.corpus import stopwords
from pyserini.search import SimpleSearcher
import spacy

import argparse
import csv
import string

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

def remove_punctuation(st):
    return " ".join(''.join(' ' if c in string.punctuation else c for c in st).split())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--anserini-index', type=str, default="indexes/dataset_collection_jsonl")
    parser.add_argument('--remove-function-words', action="store_true")
    parser.add_argument('--remove-punctuation', action="store_true")
    parser.add_argument('--lowercase-query', action="store_true")
    parser.add_argument('--remove-stopwords', action="store_true")
    args = parser.parse_args()

    searcher = SimpleSearcher(args.anserini_index)

    print("Enter search query!\n Enter \"quit\" when done")
    while True:
        print("> ")
        query = str(input())
        if query.strip() == "quit":
            break

        if args.remove_function_words:
            st = ""
            spacy_doc = nlp(query)
            for token in spacy_doc:
                if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                    st = st + " " + token.string 
            query = " ".join(st.split())
        if args.remove_punctuation:
            query = remove_punctuation(query)
        if args.lowercase_query:
            query = query.lower()
        if args.remove_stopwords:
            stopwords_without_punctuation = [remove_punctuation(stopword) for stopword in stopwords.words('english')]
            non_stopwords = []
            for w in query.split():
                if w.lower() not in stopwords.words('english'):
                    non_stopwords.append(w)
            query = " ".join(non_stopwords)
        hits = searcher.search(query)

        previous_hits = set()
        for rank, hit in enumerate(hits):
            docid = "_".join(hit.docid.split())
            if docid in previous_hits:
                rank -= 1
                continue
            else:
                previous_hits.add(docid)
            print(f"Rank {rank+1}\tScore {hit.score}\nDataset: {hit.docid}\nDescription:{hit.raw}\n\n")
        print("======================\n======================\n======================")
