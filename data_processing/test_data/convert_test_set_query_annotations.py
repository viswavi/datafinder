# python convert_test_set_query_annotations.py

import csv
from collections import defaultdict
import json
import jsonlines

comma_replacement = "‽"

if __name__ == "__main__":
    rows = []
    for row in csv.DictReader(open("data/test_set_query_annotations.csv")):
        for k in row:
            row[k] = row[k].replace(comma_replacement, ",")
        row['Bad query?'] = row['Bad query?'].lower().strip() == "yes"
        row['Contains label?'] = row['Contains label?'].lower().strip() == "yes"
        rows.append(row)
    throw_away_tldrs_mapping = {}
    for row in rows:
        query = row["Original Query"].strip()
        if query not in throw_away_tldrs_mapping:
            throw_away_tldrs_mapping[query] = False
        if throw_away_tldrs_mapping[query] is False:
            if row['Bad query?'] is True:
                throw_away_tldrs_mapping[query] = True

    json.dump(throw_away_tldrs_mapping, open("bad_tldrs_mapping.json", 'w'))
