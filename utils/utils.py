from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string

def separate_trailing_punctuation(token):
    punctuation = []
    for i, c in enumerate(reversed(token)):
        if c not in string.punctuation:
            break
        else:
            punctuation.append(c)
    punctuation_string = "".join(reversed(punctuation))
    if len(punctuation) == 0:
        return token, punctuation_string
    return token[:-i], punctuation_string

def remove_stopwords(s):
    return " ".join([t for t in s.split() if t not in stop_words])

def IOU_without_stopwords(candidate, gold):
    gold_tokens = set(gold)
    candidate_stopwords = set(candidate).intersection(stop_words)
    if len(candidate_stopwords) > 0 and candidate_stopwords not in gold_tokens:
        return 0.0
    if len(candidate) == 0 or len(gold) == 0:
        return 0.0
    if candidate in gold:
        intersection = len(candidate)
    elif gold in candidate:
        intersection = len(gold)
    else:
        intersection = 0
    union = len(candidate) + len(gold) - intersection
    IOU = float(intersection) / union
    return IOU

def scrub_dataset_references(tldr, dataset_names, replacement_token = "[DATASET]", max_ngram = 5, IOU_threshold = 0.5):
    replaced_tldr = tldr
    for n in reversed(range(1, max_ngram+1)):
        initial_tldr = replaced_tldr
        tokens = replaced_tldr.split()
        ngram_replacement_indices = []
        start_idx = 0
        while start_idx + n - 1 < len(tokens):
            ngram_tokens = tokens[start_idx:start_idx+n]
            ngram = " ".join(ngram_tokens)
            t_no_trail, _ = separate_trailing_punctuation(ngram)
            replace_token = False
            for dataset_name in dataset_names:
                IOU = IOU_without_stopwords(t_no_trail, dataset_name)
                if IOU >= IOU_threshold:
                    replace_token = True
                    break
            if replace_token:
                ngram_replacement_indices.append(start_idx)
                start_idx += start_idx + n
            else:
                start_idx += 1
        replaced_tokens = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if i in ngram_replacement_indices:
                _, trailing_punctuation = separate_trailing_punctuation(" ".join(tokens[i:i+n]))
                replaced_tokens.append( replacement_token + trailing_punctuation)
                i += n
            else:
                replaced_tokens.append(t)
                i += 1
        replaced_tldr = " ".join(replaced_tokens)

    for dataset_name in sorted(dataset_names, key=len, reverse=True):
        replaced_tldr = replaced_tldr.replace(dataset_name, replacement_token)

    replaced_tokens = []
    for token in replaced_tldr.split():
        # Filter out repeated contiguous [DATASET] tokens
        if len(replaced_tokens) > 0 and replaced_tokens[-1] == replacement_token and token == replacement_token:
            continue
        else:
            replaced_tokens.append(token)
    replaced_tldr = " ".join(replaced_tokens)
    return replaced_tldr
