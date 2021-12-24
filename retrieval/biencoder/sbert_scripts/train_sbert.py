'''
CUDA_VISIBLE_DEVICES=2  python sbert_scripts/train_sbert.py \
    --output-training-directory sbert_models/distilbert_nli_hard_negatives_lr_2e__5 \
    --base-model distilbert-base-nli-mean-tokens \
    --training-data-file tevatron_data/training_raw_hard_negatives_sep/train_data.json \
    --batch-size  \
    --lr 2e-5

CUDA_VISIBLE_DEVICES=2  python sbert_scripts/train_sbert.py \
    --output-training-directory sbert_models/distilbert_nli_hard_negatives_lr_2e__5 \
    --base-model distilroberta-base \
    --training-data-file tevatron_data/training_raw_hard_negatives_sep/train_data.json \
    --batch-size  \
    --lr 2e-5
'''

import argparse
import jsonlines
import random
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

BIENCODER="bi-encoder"
CROSSENCODER="cross-encoder"

parser = argparse.ArgumentParser()
parser.add_argument('--output-training-directory', type=str, default="sbert_models/bert_hard_negatives")
parser.add_argument('--encoder-style', choices=[BIENCODER, CROSSENCODER], default="bert-base-uncased")
parser.add_argument('--base-model', type=str, default="bert-base-uncased")
parser.add_argument('--training-data-file', type=str, default="tevatron_data/training_raw_hard_negatives_sep/train_data.json")
parser.add_argument('--eval-data-file', type=str, default="tevatron_data/training_raw_hard_negatives_sep/train_data.json")
parser.add_argument('--batch-size', type=int, default=12)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--num-epochs', type=int, default=2)
parser.add_argument('--warmup-steps', type=int, default=100)

def construct_model(base_model, encoder_style):
    # word_embedding_model = models.Transformer(base_model, max_seq_length=256)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    if encoder_style == BIENCODER:
        model = SentenceTransformer(base_model)
        train_loss = losses.CosineSimilarityLoss(model)
    elif encoder_style == CROSSENCODER:
        model = CrossEncoder(base_model, num_labels=1, max_length=512)
        train_loss = None
    return model, train_loss

def load_data(training_data_file, training_split=0.9, batch_size=16):
    train_examples = []
    validation_examples = []
    for row in jsonlines.open(training_data_file):
        query = row["query"]
        for positive in row["positives"]:
            sample = InputExample(texts=[query, positive], label=1.0)
            if random.random() < training_split:
                train_examples.append(sample)
            else:
                validation_examples.append(sample)
        for negative in row["negatives"]:
            sample = InputExample(texts=[query, negative], label=0.0)
            if random.random() < training_split:
                train_examples.append(sample)
            else:
                validation_examples.append(sample)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    return train_dataloader, validation_examples

if __name__ == "__main__":
    args = parser.parse_args()
    model, train_loss = construct_model(args.base_model, args.encoder_style)
    train_dataloader, validation_examples = load_data(args.training_data_file, batch_size=args.batch_size)
    if args.encoder_style == BIENCODER:
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(validation_examples)
    else:
        evaluator = CECorrelationEvaluator.from_input_examples(validation_examples)
    if args.encoder_style == BIENCODER:
        model.fit(train_objectives=[(train_dataloader, train_loss)],
                epochs=args.num_epochs,
                warmup_steps=args.warmup_steps,
                evaluator=evaluator,
                evaluation_steps=500,
                optimizer_params= { 'lr': args.lr },
                output_path=args.output_training_directory,
                use_amp=True)
    elif args.encoder_style == CROSSENCODER:
        model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=args.num_epochs,
          warmup_steps=args.warmup_steps,
          evaluation_steps=500,
          optimizer_params= { 'lr': args.lr },
          output_path=args.output_training_directory)

    evaluator(model)
