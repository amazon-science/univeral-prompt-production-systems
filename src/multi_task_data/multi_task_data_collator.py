from typing import Dict
import dataclasses
from dataclasses import dataclass

import torch
from transformers.data.data_collator import DataCollator, DataCollatorForSeq2Seq


@dataclass
class MultiTaskDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            # MBart had different decoder_start_token_id depending on the language and
            # decoder_start_token_id during training for multilanguage batches
            if getattr(features, "decoder_start_token_id", None) is not None and self.model.training:
                self.model.config.decoder_start_token_id = features["decoder_start_token_id"]
            elif hasattr(features, 'decoder_start_token_id'):
                self.model.config.decoder_start_token_id = features["decoder_start_token_id"][0].item()
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features