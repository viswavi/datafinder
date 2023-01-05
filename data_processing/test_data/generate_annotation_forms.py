# python generate_annotation_forms.py

from __future__ import print_function

from apiclient import discovery
import argparse
import json
import jsonlines
from httplib2 import Http
from oauth2client import client, file, tools
import os

parser = argparse.ArgumentParser()
parser.add_argument("--test-set-queries", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_abstracts.temp")
parser.add_argument("--tldrs", default="/Users/vijay/Documents/code/dataset-recommendation/scirex_tldrs.hypo")
parser.add_argument("--form-ids-directory", default="/Users/vijay/Documents/code/dataset-recommendation/data_processing/test_data/")
parser.add_argument("--gpt3-suggestions", default="/Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_gpt3_postprocessed.jsonl")
parser.add_argument("--galactica-suggestions", default="/Users/vijay/Documents/code/dataset-recommendation/test_abstracts_parsed_by_galactica_postprocessed.jsonl")
parser.add_argument("--batch-tag", default="Z")
parser.add_argument("--chunk-idxs-to-generate", default=None)
parser.add_argument("--num-forms-to-generate", type=int, default=5)

def load_text_chunks(chunk_sizes, offer_paid_option, abstracts, tldrs, gpt3_suggestions, galactica_suggestions, index_offset=0):
    assert len(abstracts) == len(tldrs)
    chunks = [{"offer_paid_option": offer_paid_option[0], "docs": []}]
    for i, (a, t,  gpt, galactica) in enumerate(zip(abstracts, tldrs, gpt3_suggestions, galactica_suggestions)):
        text_dict = {"abstract": a,
                     "tldr": t,
                     "gpt_suggestions": gpt,
                     "galactica_suggestions": galactica,
                     "test_set_idx": i + index_offset}
        chunks[-1]["docs"].append(text_dict)
        if len(chunks[-1]["docs"]) == chunk_sizes[0]:
            chunk_sizes = chunk_sizes[1:]
            if len(chunk_sizes) == 0:
                break
            offer_paid_option = offer_paid_option[1:]
            chunks.append({"offer_paid_option": offer_paid_option[0], "docs": []})    
    return chunks

def construct_annotation_instance_update(abstract, tldr, gpt3_suggestions, galactica_suggestions, paper_index=1):
    galactica_task_suggestion = galactica_suggestions["Task"]
    galactica_modality_suggestion = galactica_suggestions["Modality"]
    if galactica_modality_suggestion == "N/A":
        galactica_modality_suggestion = "N/A (leave empty)"
    if galactica_modality_suggestion == "images":
        galactica_modality_suggestion = "Image"
    if galactica_modality_suggestion == "videos":
        galactica_modality_suggestion = "Video"
    if galactica_modality_suggestion == "video":
        galactica_modality_suggestion = "Video"
    if galactica_modality_suggestion == "text":
        galactica_modality_suggestion = "Text"
    galactica_domain_suggestion = galactica_suggestions["Domain"]
    if galactica_domain_suggestion == "N/A":
        galactica_domain_suggestion = "N/A (leave empty)"
    galactica_training_style_suggestion = galactica_suggestions["Training Style"]
    if galactica_training_style_suggestion in ["fully supervised", "fully-supervised", "supervised"]:
        galactica_training_style_suggestion = "Supervised training/finetuning"
    elif galactica_training_style_suggestion == "N/A":
        galactica_training_style_suggestion = "N/A (leave empty)"
    else:
        galactica_training_style_suggestion = galactica_training_style_suggestion.capitalize()
    galactica_text_length_suggestion = galactica_suggestions["Text Length"]
    if galactica_text_length_suggestion == "N/A":
        galactica_text_length_suggestion = "N/A (leave empty)"
    else:
        galactica_text_length_suggestion = galactica_text_length_suggestion.capitalize()
    galactica_language_suggestion = galactica_suggestions["Language Required"]
    if galactica_language_suggestion == "N/A":
        galactica_language_suggestion = "N/A (leave empty)"
    galactica_summary_suggestion = galactica_suggestions["Single-Sentence Summary"]

    gpt3_task_suggestion = gpt3_suggestions["Task"]
    gpt3_modality_suggestion = gpt3_suggestions["Modality"]
    if gpt3_modality_suggestion == "N/A":
        gpt3_modality_suggestion = "N/A (leave empty)"
    if gpt3_modality_suggestion == "images":
        gpt3_modality_suggestion = "Image"
    if gpt3_modality_suggestion == "videos":
        gpt3_modality_suggestion = "Video"
    if gpt3_modality_suggestion == "video":
        gpt3_modality_suggestion = "Video"
    if gpt3_modality_suggestion == "text":
        gpt3_modality_suggestion = "Text"
    gpt3_domain_suggestion = gpt3_suggestions["Domain"]
    if gpt3_domain_suggestion == "N/A":
        gpt3_domain_suggestion = "N/A (leave empty)"
    gpt3_training_style_suggestion = gpt3_suggestions["Training Style"]
    if gpt3_training_style_suggestion in ["fully supervised", "fully-supervised", "supervised"]:
        gpt3_training_style_suggestion = "Supervised training/finetuning"
    elif gpt3_training_style_suggestion == "N/A":
        gpt3_training_style_suggestion = "N/A (leave empty)"
    else:
        gpt3_training_style_suggestion = gpt3_training_style_suggestion.capitalize()
    gpt3_text_length_suggestion = gpt3_suggestions["Text Length"]
    if gpt3_text_length_suggestion == "N/A":
        gpt3_text_length_suggestion = "N/A (leave empty)"
    else:
        gpt3_text_length_suggestion = gpt3_text_length_suggestion.capitalize()
    gpt3_language_suggestion = gpt3_suggestions["Language Required"]
    if gpt3_language_suggestion == "N/A":
        gpt3_language_suggestion = "N/A (leave empty)"
    gpt3_summary_suggestion = gpt3_suggestions["Final Summary"]

    update_requests = [{
            "createItem": {
                "item": {
                    'title': f'Paper #{paper_index}',
                    'description': f'Abstract:\n{abstract}\n\nTLDR:\n{tldr}',
                    "pageBreakItem": {
                    }
                },
                "location": {
                    "index": (paper_index-1)*8+2
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Task(s) mentioned in paper',
                    "description": f'Provide comma-delimited list if multiple tasks are mentioned.\n(if no task is mentioned, leave this field empty, skip all the fields below, and skip to the next paper)\n\nSuggestion from GPT-3:\t\t{gpt3_task_suggestion}\nSuggestion from Galactica:\t{galactica_task_suggestion}',
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
                    'title': 'Domain of paper (e.g. biomedical or aerial)',
                    "description": f'Provide comma-delimited list if multiple tasks are mentioned. Leave empty if no task is mentioned (or clearly implied).\n\nSuggestion from GPT-3:\t\t{gpt3_domain_suggestion}\nSuggestion from Galactica:\t{galactica_domain_suggestion}',
                    "questionItem": {
                        "question": {
                            'textQuestion': {}
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
                    'title': 'Modality required by paper',
                    "description": f'Leave empty if not specified (or clearly implied).\n\nSuggestion from GPT-3:\t\t{gpt3_modality_suggestion}\nSuggestion from Galactica:\t{galactica_modality_suggestion}',
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
                    "index": (paper_index-1)*8+5
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Language of data or labels required',
                    "description": f'Do not select anything if this is not specified (or clearly implied).\n\nSuggestion from GPT-3:\t\t{gpt3_language_suggestion}\nSuggestion from Galactica:\t{galactica_language_suggestion}',
                    "questionItem": {
                        "question": {
                            'textQuestion': {}
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
                    'title': 'Training style',
                    "description": f'Do not select anything if this is not specified (or clearly implied).\n\nSuggestion from GPT-3:\t\t{gpt3_training_style_suggestion}\nSuggestion from Galactica:\t{galactica_training_style_suggestion}',
                    "questionItem": {
                        "question": {
                            'choiceQuestion': {
                                'type': 'RADIO',
                                'options': [{
                                    'value': 'Supervised training/finetuning'
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
                    "index": (paper_index-1)*8+7
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Does this paper discuss sentence-level or paragraph-level text processing?',
                    "description": f'Only applicable if this is an NLP paper. Do not select anything if this is not specified (or clearly implied).\n\nSuggestion from GPT-3:\t\t{gpt3_text_length_suggestion}\nSuggestion from Galactica:\t{galactica_text_length_suggestion}',
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
                    "index": (paper_index-1)*8+8
                }
            }
        }, {
            "createItem": {
                "item": {
                    'title': 'Final Query',
                    'description': f'Use any information you extracted in the questions above, along with any other critical information from the abstract into a single-sentence summary that might inform the choice of dataset.\n\nSuggestion from GPT-3:\t\t"{gpt3_summary_suggestion}"\nSuggestion from Galactica:\t"{galactica_summary_suggestion}"',
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
                    "index": (paper_index-1)*8+9
                }
            }
        }
    ]
    return update_requests

if __name__ == "__main__":
    args = parser.parse_args()

    form_index_directory = os.path.join(args.form_ids_directory, f"batch_{args.batch_tag}")
    os.makedirs(form_index_directory)

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

    example_tldr_1 = tldrs[0]
    example_tldr_2 = tldrs[1]

    chunk_sizes = [10] * 2 + [15] * 4 + [10] * 36
    offer_paid_option = [False, False, True, True, True, False] + [True] * 13 + [False] * (42 - 19)

    if args.num_forms_to_generate is not None:
        text_chunks = load_text_chunks(chunk_sizes[:args.num_forms_to_generate], offer_paid_option, abstracts, tldrs, gpt3_suggestions, galactica_suggestions)
    else:
        text_chunks = load_text_chunks(chunk_sizes, offer_paid_option, abstracts, tldrs, gpt3_suggestions, galactica_suggestions)

    SCOPES = "https://www.googleapis.com/auth/drive"
    DISCOVERY_DOC = "https://forms.googleapis.com/$discovery/rest?version=v1"

    store = file.Storage('token.json')
    creds = None
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secrets.json', SCOPES)
        creds = tools.run_flow(flow, store)

    form_service = discovery.build('forms', 'v1', http=creds.authorize(
        Http()), discoveryServiceUrl=DISCOVERY_DOC, static_discovery=False)

    #example_string_1 = f"Let's go through an example:\n\nAbstract: {example_abstract_1}\n\nTLDR: {example_tldr_1}\n\n*Things you would need to fill out*:\n\nTask(s) mentioned in paper:\nsemantic image segmentation\n\nDomain of paper (e.g. biomedical or aerial):\nautonomous driving \n\nModality required by paper:\nimages\n\nLanguage of data or labels required:\nN/A\n\nTraining style:\nlarge-scale supervised training\n\nDoes this paper discuss sentence-level or paragraph-level text processing?:\nN/A\n\nFinal Query:\nthere are multiple queries you could write for this abstract:\n    -We want to build a system for semantic image segmentation for self-driving cars\n    -I propose a method to improve semantic image segmentation from images using large-scale supervised learning\n    -I want to work on semantic image segmentation for autonomous vehicles"
    #example_string_2 = f"Example #2:\n\n*Abstract*: {example_abstract_2}\n\n*TLDR*: {example_tldr_2}\n\n*Task(s) mentioned in paper*: 3D image segmentation\n*Domain of paper (e.g. biomedical or aerial)*: medical images\n*Modality required by paper*: 3D images\n*Language of data or labels required*: N/A\n*Training style*:  large-scale supervised training\n*Does this paper discuss sentence-level or paragraph-level text processing?*:  N/A\n*Final Query*:\nThere are multiple possible queries for this abstract, including:\n    -We propose an approach for 3D image segmentation for biomedical image analysis\n    -We want to develop an efficient CNN-based model for 3D image segmentation from biomedical 3D images\n    -We propose an approach for 3D image segmentation from 3D (volumetric) images from fMRI scans"

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

        annotation_prompt = """Given an abstract from an artificial intelligence paper:
1) extract keyphrases regarding:
    - the task (e.g. image classification)
    - data modality (e.g. images or speech)
    - domain (e.g. biomedical or aerial)
    - training style (unsupervised, semi-supervised, supervised, or reinforcement learning)
    - text length (sentence-level or paragraph-level)
    - language required (e.g. English)
2) write a very short, single-sentence summary that contains these relevant keyphrases, only including other information if critical to understanding the abstract.

Things to keep in mind:
    - Do not spend more than *2 minutes* in total on each example. If you find yourself taking too long to understand or tag a given abstract, just skip to the next one.
    - Do not mention any datasets by name.
----------------

Let's go through an example:
Abstract: Semantic image segmentation is an essential component of modern autonomous driving systems, as an accurate understanding of the surrounding scene is crucial to navigation and action planning. Current state-of-the-art approaches in semantic image segmentation rely on pre-trained networks that were initially developed for classifying images as a whole. While these networks exhibit outstanding recognition performance (i.e., what is visible?), they lack localization accuracy (i.e., where precisely is something located?). Therefore, additional processing steps have to be performed in order to obtain pixel-accurate segmentation masks at the full image resolution. To alleviate this problem we propose a novel ResNet-like architecture that exhibits strong localization and recognition performance. We combine multi-scale context with pixel-level accuracy by using two processing streams within our network: One stream carries information at the full image resolution, enabling precise adherence to segment boundaries. The other stream undergoes a sequence of pooling operations to obtain robust features for recognition. The two streams are coupled at the full image resolution using residuals. Without additional processing steps and without pre-training, our approach achieves an intersection-over-union score of 71.8% on the Cityscapes dataset.

TLDR: We propose a novel ResNet-like architecture that combines multi-scale context with pixel-level accuracy for Semantic Image Segmentation.
---
*Things you would need to fill out*:

Task(s) mentioned in paper:
semantic image segmentation

Domain of paper (e.g. biomedical or aerial):
autonomous driving

Modality required by paper:
images

Language of data or labels required:
N/A

Training style:
supervised

Does this paper discuss sentence-level or paragraph-level text processing?:
N/A

Final Query:
(there are multiple queries you could write for this abstract)

    -We want to build a system for semantic image segmentation for self-driving cars
    -I propose a method to improve semantic image segmentation from images using large-scale supervised learning
    -I want to work on semantic image segmentation for autonomous vehicles
"""

        crediting_options = [{
                                        'value': '$10 Amazon Gift Card'
                                    }, {
                                        'value': 'Acknowledgement in our paper (to be submitted to ACL 2023)'
                                    }, {
                                        'value': 'Group lunch on me (if you know the lead author, Vijay, personally)'
                                    }, {
                                        'value': 'None of the above.'
                                    }
                            ]
        if chunk["offer_paid_option"] is False:
            crediting_options = crediting_options[1:]

        update = {
            "requests": [{
                "updateFormInfo": {
                    "info": {
                        'description': annotation_prompt,
                    },
                    "updateMask": "description"
                }},
            {
                "createItem": {
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
            },
            {
                "createItem": {
                    "item": {
                        'title': 'How would you prefer to be credited for your effort?',
                        "questionItem": {
                            "question": {
                                'choiceQuestion': {
                                    'type': 'RADIO',
                                    'options': crediting_options
                                }
                            }
                        }
                    },
                    "location": {
                        "index": 1
                    }
                }
        }]
        }

        for i, doc in enumerate(chunk["docs"]):
            update["requests"].extend(construct_annotation_instance_update(doc["abstract"], doc["tldr"], doc["gpt_suggestions"], doc["galactica_suggestions"], paper_index=i+1))


        updateResult = form_service.forms().batchUpdate(
            formId=created_form_id, body=update).execute()

        print(updateResult)
        print(f"Created form ID: {created_form_id}")
        chunk_info = {
            "idx": chunk_idx + 1,
            "form_id": created_form_id,
            "abstract_info": chunk["docs"]
        }
        json.dump(chunk_info, open(os.path.join(form_index_directory, f"{chunk_idx + 1}.json"), 'w'), indent=4)