# python generate_annotation_form.py

from __future__ import print_function

from apiclient import discovery
import argparse
import json
from httplib2 import Http
from oauth2client import client, file, tools

def load_text_chunks(abstracts, tldrs, chunk_size=20):
    assert len(abstracts) == len(tldrs)
    chunks = [[]]
    for i, (a, t) in enumerate(zip(abstracts, tldrs)):
        text_dict = {"abstract": a, "tldr": t}
        chunks[-1].append(text_dict)
        if len(chunks[-1]) == chunk_size:
            chunks.append([])
    return chunks

def construct_annotation_instance_update(abstract, tldr, paper_index=1):
    update_requests = [{
            "createItem": {
                "item": {
                    'title': f'Paper #{paper_index}',
                    'description': f'**Abstract**:\n{abstract}\n\n*Machine-Generated Summary*:\n{tldr}',
                    "pageBreakItem": {
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+1
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Task(s) mentioned in paper',
                    "description": "Provide comma-delimited list if multiple tasks are mentioned. Leave empty if no task is mentioned.",
                    "questionItem": {
                        "question": {
                            'textQuestion': {}
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+2
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Domain of paper (e.g. biomedical or aerial)',
                    "description": "Provide comma-delimited list if multiple tasks are mentioned. Leave empty if no task is mentioned.",
                    "questionItem": {
                        "question": {
                            'textQuestion': {}
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+3
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Modality required by paper',
                    "description": "Leave empty if not specified.",
                    "questionItem": {
                        "question": {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [{
                                    'value': 'Text'
                                }, {
                                    'value': 'Image'
                                }, {
                                    'value': 'Video'
                                }, {
                                    'isOther': True
                                }]
                            }
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+4
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Language of data or labels required',
                    "description": "Do not select anything if this is not specified.",
                    "questionItem": {
                        "question": {
                            'textQuestion': {}
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+5
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Training data requirements',
                    "description": "Do not select anything if this is not specified.",
                    "questionItem": {
                        "question": {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [{
                                    'value': 'Large-scale labeled data'
                                }, {
                                    'value': 'Few-shot'
                                }, {
                                    'value': 'Unsupervised'
                                }, {
                                    'isOther': True
                                }]
                            }
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+6
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Does this paper discuss sentence-level or paragraph-level text processing?',
                    "description": "Only applicable if this is an NLP paper. Do not select anything if this is not specified.",
                    "questionItem": {
                        "question": {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [{
                                    'value': 'Sentence-level'
                                }, {
                                    'value': 'Paragraph-level'
                                }, {
                                    'isOther': True
                                }]
                            }
                        }
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+7
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Final Query',
                    'description': 'Use any information you extracted in the questions above, along with any other critical information from the abstract that might inform the choice of dataset.',
                    "questionItem": {
                        "question": {
                            'required': True,
                            'textQuestion': {
                                'paragraph': True
                            }
                        },
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+8
                }
            }
        }
    ]
    return update_requests

parser = argparse.ArgumentParser()
parser.add_argument("--test-set-queries", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_abstracts.temp")
parser.add_argument("--tldrs", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_tldrs.hypo")
parser.add_argument("--form-ids-file", default="/Users/vijay/Documents/code/dataset-recommendation/data_processing/test_data/annotation_form_tracker.json")

if __name__ == "__main__":
    args = parser.parse_args()

    abstracts = []
    for ab in open(args.test_set_queries).read().split("\n"):
        if len(ab.strip()) == 0:
            continue
        if ab.endswith(" <|TLDR|> ."):
            ab = ab[:-len(" <|TLDR|> .")]
        abstracts.append(ab)
    tldrs = []
    for t in open(args.tldrs).read().split("\n"):
        if len(t.strip()) > 0:
            tldrs.append(t)
    text_chunks = load_text_chunks(abstracts, tldrs)

    SCOPES = "https://www.googleapis.com/auth/drive"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    form_ids = []
    for chunk_idx, chunk in enumerate(text_chunks):
        initialForm = {
            "info": {
                'title': f'DatasetFinder Annotation Form #{chunk_idx + 1}',
                'documentTitle': f'(#{chunk_idx + 1}) DatasetFinder Annotation Form',
            },
        }
        # Prints the details of the sample form
        createResult = form_service.forms().create(body=initialForm).execute()
        created_form_id = createResult["formId"]

        update = {
            "requests": [{
                "updateFormInfo": {
                    "info": {
                        'description': 'We are developing a dataset search engine which accepts natural language descriptions of what the user wants to build.\nFor each abstract, extract the following keyphrases:\n- task\n- domain (if necessary)\n- modality required (if necessary)\n- language of data or labels required (if necessary)\n- training data requirements: unsupervised, few shot, or large scale (if necessary)\n- sentence-level or paragraph-level (if necessary)\n\nThen, for each abstract, write a query for our search engine that the authors of the paper could have used to find their datasets, incorporating the keyphrases above to the extent that it is natural. Write the system description in the first person future tense (as if you were the abstract author of the provided abstract).\n\nDo not mention any datasets by name.',
                    },
                    "updateMask": "description"
                }},
                {"createItem": {
                    "item": {
                        'title': "What's your Andrew ID?",
                        "description": "Enter your email address if you are not a CMU affiliate.",
                        "questionItem": {
                            "question": {
                                'required': True,
                                'textQuestion': {}
                            }
                        }
                    },
                    "location": {
                        "index": 0
                    }
                }
            }]
        }

        for i, doc in enumerate(chunk):
            update["requests"].extend(construct_annotation_instance_update(doc["abstract"], doc["tldr"], paper_index=i+1))


        updateResult = form_service.forms().batchUpdate(
            formId=created_form_id, body=update).execute()

        print(updateResult)
        print(f"Created form ID: {created_form_id}")
        chunk_info = {
            "idx": chunk_idx,
            "form_id": created_form_id,
            "abstract_ifo": chunk
        }
        form_ids.append(chunk_info)
        break
    json.dump(form_ids, open(args.form_ids_file, 'w'), indent=4)