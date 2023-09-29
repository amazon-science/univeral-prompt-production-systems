import nltk
import numpy as np
from typing import Callable, Dict
from datasets import load_metric
from transformers import EvalPrediction
from rouge_score import rouge_scorer, scoring
from src.multi_task_data.utils import DatasetName, TaskName

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

PAD_TOKEN_ID = -100

rouge_metric = load_metric("rouge")
meteor_metric = load_metric("meteor")
bleu_metric = load_metric('sacrebleu')


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels


def postprocess_translation_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def postprocess_summarization_text(preds, labels):
    preds, labels = postprocess_text(preds, labels)
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_all_summarization_metrics(predictions, tokenizer, get_bleurt=False):

    preds, labels = predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != PAD_TOKEN_ID, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    #decoded_preds, decoded_labels = postprocess_summarization_text(decoded_preds, decoded_labels)
    decoded_preds = list(map(str.strip, decoded_preds))
    decoded_labels = list(map(str.strip, decoded_labels))

    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {key: value.mid.fmeasure * 100 for key, value in rouge_result.items()}

    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_result = {key: value * 100 for key, value in meteor_result.items()}

    result = {**rouge_result, **meteor_result}
    if get_bleurt:
        bleurt_result = bleurt_metric.compute(predictions=decoded_preds, references=decoded_labels)
        bleurt_result = {'bleurt': np.mean(bleurt_result['scores'])}
        result = {**result, **bleurt_result}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def compute_textgen_metrics(predictions, tokenizer):

    preds, labels = predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != PAD_TOKEN_ID, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_translation_text(decoded_preds, decoded_labels)

    result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def build_compute_metrics_fn(
    task_name, tokenizer
) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(prediction: EvalPrediction):
        if task_name == TaskName.xsum_summarize.value.lower() \
                or task_name == TaskName.xsum_extractive.value.lower() \
                or task_name == TaskName.cnndm_topic_summarize.value.lower() \
                or task_name == TaskName.cnndm_topic_extractive.value.lower():
            return compute_all_summarization_metrics(prediction, tokenizer)
        elif task_name == TaskName.newsqa.value.lower() \
                or task_name == TaskName.xsum_paraphrase.value.lower() \
                or task_name == TaskName.xsum_ner.value.lower() \
                or TaskName.translation.value.lower() in task_name:
            return compute_textgen_metrics(prediction, tokenizer)
        else:
            raise NotImplementedError(
                f"Dataset '{task_name}' handling not implemented"
            )
    return compute_metrics_fn

