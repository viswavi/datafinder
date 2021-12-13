# python scrape_google_dataset_search.py test_set_query_annotations.csv google_search_results.tsv

import argparse
from bs4 import BeautifulSoup
import csv
import random
import requests
import time
from tqdm import tqdm

def get_url_requests(query):
    query_string = "%20".join(query.split())
    url = f"https://datasetsearch.research.google.com/search?query={query_string}%20from%3Apaperswithcode.com"
    session = requests.Session()
    response = session.get(url)
    if not response.ok or response.status_code != 200:
        print(f"Error received: {response.reason}")
        breakpoint()
    return response.text

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    datasets = []
    result_elements = soup.find_all("li", {"class": "UnWQ5"})
    for element in result_elements:
        sub_datasets = element.find_all("h1", "iKH1Bc")
        if len(sub_datasets) == 0:
            continue
        assert len(sub_datasets) == 1, breakpoint()
        dataset = sub_datasets[0].text
        dataset_tokens = dataset.split()
        if dataset_tokens[-1] == "Dataset":
            dataset = " ".join(dataset_tokens[:-1])
        if dataset.startswith("Data from: "):
            dataset = dataset[len("Data from: "):]
        datasets.append(dataset)
        print(f"dataset: {dataset}")

    return datasets
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_set_query_annotations', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    full_queries = []
    keyword_queries = []
    for row in csv.DictReader(open(args.test_set_query_annotations)):
        full_queries.append(row['Original Query'])
        keyword_queries.append(row['Keywords'])

    results = []
    for q in tqdm(full_queries):
        q = q.strip()
        html = get_url_requests(q)
        datasets = parse_html(html)
        results.append(datasets)
        sleeptime = random.randint(50, 150)/100.0
        time.sleep(sleeptime)

    results_tsv_contents = [full_query + "\t" + keywords + "\t" + "\t".join(row) for (full_query, keywords, row) in zip(full_queries, keyword_queries, results)]

    outfile = open(args.output_file, 'w')
    outfile.write("\n".join(results_tsv_contents))
    outfile.close()