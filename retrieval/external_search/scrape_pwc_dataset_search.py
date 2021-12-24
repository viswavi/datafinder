# python scrape_pwc_dataset_search.py test_set_query_annotations.csv pwc_search_results.tsv

import argparse
from bs4 import BeautifulSoup
import csv
import random
import requests
import time
from tqdm import tqdm

def get_url_requests(query):
    query_string = "+".join(query.split())
    url = f"https://paperswithcode.com/datasets?q={query_string}&v=lst&o=match"
    session = requests.Session()
    response = session.get(url)
    if not response.ok or response.status_code != 200:
        print(f"Error received: {response.reason}")
        breakpoint()
    return response.text

def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    datasets = []
    result_elements = soup.find_all("span", {"class": "name"})
    for element in result_elements:
        dataset = list(element.children)[0].string.strip()
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
        sleeptime = random.randint(100, 500)/100.0
        time.sleep(sleeptime)

    results_tsv_contents = ["\t".join(row) for row in results]

    outfile = open(args.output_file, 'w')
    outfile.write("\n".join(results_tsv_contents))
    outfile.close()