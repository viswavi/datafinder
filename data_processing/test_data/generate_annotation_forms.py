# python generate_annotation_form.py

from __future__ import print_function

from apiclient import discovery
import argparse
import json
import jsonlines
from httplib2 import Http
from oauth2client import client, file, tools

def load_text_chunks(abstracts, tldrs, gpt3_suggestions, galactica_suggestions, chunk_size=20, num_small_chunks=4, small_chunk_size=5, SMALL_CHUNKS_ONLY=False):
    assert len(abstracts) == len(tldrs)
    chunks = [[]]
    small_chunks = 0
    for i, (a, t,  gpt_suggestions, galactica_suggestions) in enumerate(zip(abstracts, tldrs, gpt3_suggestions, galactica_suggestions)):
        text_dict = {"abstract": a,
                     "tldr": t,
                     "gpt_suggestions": gpt_suggestions,
                     "galactica_suggestions": galactica_suggestions}
        chunks[-1].append(text_dict)
        if small_chunks < num_small_chunks:
            if len(chunks[-1]) == small_chunk_size:
                chunks.append([])
                small_chunks += 1
        else:
            if SMALL_CHUNKS_ONLY:
                break
            if len(chunks[-1]) == chunk_size:
                chunks.append([])
    return chunks

def construct_annotation_instance_update(abstract, tldr, gpt3_suggestions, galactica_suggestions, paper_index=1):
    galactica_task_suggestion = galactica_suggestions["Task"]
    galactica_modality_suggestion = galactica_suggestions["Modality"].capitalize()
    galactica_domain_suggestion = galactica_suggestions["Domain"]
    galactica_training_style_suggestion = galactica_suggestions["Training Style"]
    if galactica_training_style_suggestion == "fully supervised":
        galactica_training_style_suggestion = "Large-scale supervised training/finetuning"

    galactica_text_length_suggestion = galactica_suggestions["Text Length"]
    galactica_language_suggestion = galactica_suggestions["Language Required"]
    galactica_summary_suggestion = galactica_suggestions["Single-Sentence Summary"]

    gpt3_task_suggestion = gpt3_suggestions["Task"]
    gpt3_modality_suggestion = gpt3_suggestions["Modality"].capitalize()
    gpt3_domain_suggestion = gpt3_suggestions["Domain"]
    gpt3_training_style_suggestion = gpt3_suggestions["Training Style"]
    if gpt3_training_style_suggestion == "fully supervised":
        gpt3_training_style_suggestion = "Large-scale supervised training/finetuning"
    gpt3_text_length_suggestion = gpt3_suggestions["Text Length"]
    gpt3_language_suggestion = gpt3_suggestions["Language Required"]
    gpt3_summary_suggestion = gpt3_suggestions["Final Summary"]

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
                    "description": f'Provide comma-delimited list if multiple tasks are mentioned.\n**If no task is mentioned, leave this field empty, skip all the fields below, and skip to the next paper.**\n\nSuggestion from GPT-3: "{gpt3_task_suggestion}"\nSuggestion from Galactica: "{galactica_task_suggestion}"',
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
                    "description": f'Provide comma-delimited list if multiple tasks are mentioned. Leave empty if no task is mentioned.\n\nSuggestion from GPT-3: "{gpt3_domain_suggestion}"\nSuggestion from Galactica: "{galactica_domain_suggestion}"',
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
                    "description": f'Leave empty if not specified.\n\nSuggestion from GPT-3: "{gpt3_modality_suggestion}"\nSuggestion from Galactica: "{galactica_modality_suggestion}"',
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
                    "description": f'Do not select anything if this is not specified.\n\nSuggestion from GPT-3: "{gpt3_language_suggestion}"\nSuggestion from Galactica: "{galactica_language_suggestion}"',
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
                    'title': 'Training style',
                    "description": f'Do not select anything if this is not specified.\n\nSuggestion from GPT-3: "{gpt3_training_style_suggestion}"\nSuggestion from Galactica: "{galactica_training_style_suggestion}"',
                    "questionItem": {
                        "question": {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [{
                                    'value': 'Large-scale supervised training/finetuning'
                                }, {
                                    'value': 'Few-shot'
                                }, {
                                    'value': 'Semi-supervised'
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
                    "description": f'Only applicable if this is an NLP paper. Do not select anything if this is not specified.\n\nSuggestion from GPT-3: "{gpt3_text_length_suggestion}"\nSuggestion from Galactica: "{galactica_text_length_suggestion}"',
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
                    'description': f'Use any information you extracted in the questions above, along with any other critical information from the abstract that might inform the choice of dataset.\n\nSuggestion from GPT-3: "{gpt3_summary_suggestion}"\nSuggestion from Galactica: "{galactica_summary_suggestion}"',
                    "questionItem": {
                        "question": {
                            'required': False,
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
    breakpoint()
    return update_requests

parser = argparse.ArgumentParser()
parser.add_argument("--test-set-queries", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_abstracts.temp")
parser.add_argument("--tldrs", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_tldrs.hypo")
parser.add_argument("--form-ids-file", default="/Users/vijay/Documents/code/dataset-recommendation/data_processing/test_data/annotation_form_tracker.json")
parser.add_argument("--gpt3-suggestions", default="/Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_gpt3_postprocessed.jsonl")
parser.add_argument("--galactica-suggestions", default="/Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_galactica_postprocessed.jsonl")
parser.add_argument("--num-forms-to-generate", type=int, default=1)

if __name__ == "__main__":
    args = parser.parse_args()

    abstracts = []
    for ab in [t.strip() for t in open(args.test_set_queries).readlines()]:
        if len(ab.strip()) == 0:
            continue
        if ab.endswith(" <|TLDR|> ."):
            ab = ab[:-len(" <|TLDR|> .")]
        abstracts.append(ab)
    tldrs = []
    for t in open(args.tldrs).read().split("\n"):
        if len(t.strip()) > 0:
            tldrs.append(t)

    gpt3_suggestions = list(jsonlines.open(args.gpt3_suggestions))
    galactica_suggestions = list(jsonlines.open(args.galactica_suggestions))

    example_abstract_1 = abstracts[0]
    example_abstract_2 = abstracts[1]
    abstracts = abstracts[2:]

    example_tldr_1 = tldrs[0]
    example_tldr_2 = tldrs[1]
    tldrs = tldrs[2:]
    if args.num_forms_to_generate is not None:
        text_chunks = load_text_chunks(abstracts, tldrs, gpt3_suggestions, galactica_suggestions, num_small_chunks=args.num_forms_to_generate, SMALL_CHUNKS_ONLY=True)
    else:
        text_chunks = load_text_chunks(abstracts, tldrs, gpt3_suggestions, galactica_suggestions)

    SCOPES = "https://www.googleapis.com/auth/drive"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    example_string_1 = f"Let's go through an example:\n\nAbstract: {example_abstract_1}\n\nTLDR: {example_tldr_1}\n\n*Things you would need to fill out*:\n\nTask(s) mentioned in paper:\nsemantic image segmentation\n\nDomain of paper (e.g. biomedical or aerial):\nautonomous driving \n\nModality required by paper:\nimages\n\nLanguage of data or labels required:\nN/A\n\nTraining style:\nlarge-scale supervised training\n\nDoes this paper discuss sentence-level or paragraph-level text processing?:\nN/A\n\nFinal Query:\nthere are multiple queries you could write for this abstract:\n    -We want to build a system for semantic image segmentation for self-driving cars\n    -I propose a method to improve semantic image segmentation from images using large-scale supervised learning\n    -I want to work on semantic image segmentation for autonomous vehicles"
    #example_string_2 = f"Example #2:\n\n*Abstract*: {example_abstract_2}\n\n*TLDR*: {example_tldr_2}\n\n*Task(s) mentioned in paper*: 3D image segmentation\n*Domain of paper (e.g. biomedical or aerial)*: medical images\n*Modality required by paper*: 3D images\n*Language of data or labels required*: N/A\n*Training style*:  large-scale supervised training\n*Does this paper discuss sentence-level or paragraph-level text processing?*:  N/A\n*Final Query*:\nThere are multiple possible queries for this abstract, including:\n    -We propose an approach for 3D image segmentation for biomedical image analysis\n    -We want to develop an efficient CNN-based model for 3D image segmentation from biomedical 3D images\n    -We propose an approach for 3D image segmentation from 3D (volumetric) images from fMRI scans"

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
                        'description': f'We are developing a dataset search engine which accepts natural language descriptions of what the user wants to build.\nFor each abstract, extract the following keyphrases:\n- task\n- domain (if necessary)\n- modality required (if necessary)\n- language of data or labels required (if necessary)\n- training data requirements: unsupervised, semi-supervised, few shot, or large-scale supervised training/finetuning (if necessary)\n- sentence-level or paragraph-level (if necessary)\n\nThen, for each abstract, write a query for our search engine that the authors of the paper could have used to find their datasets, incorporating the keyphrases above to the extent that it is natural. Write the system description in the first person future tense (as if you were the abstract author of the provided abstract).\n\nDo not mention any datasets by name.\n---------\n\n{example_string_1}',
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
            update["requests"].extend(construct_annotation_instance_update(doc["abstract"], doc["tldr"], doc["gpt_suggestions"], doc["galactica_suggestions"], paper_index=i+1))


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

    json.dump(form_ids, open(args.form_ids_file, 'w'), indent=4)