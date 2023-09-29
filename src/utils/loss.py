from enum import Enum

import torch.nn as nn


class CrossEntropyLossWrapper:
    def __init__(self, num_labels):
        self.__num_labels = num_labels
        self.__loss_fct = nn.CrossEntropyLoss()

    def calculate_loss(self, logits, labels):
        return self.__loss_fct(
            logits.view(-1, self.__num_labels), labels.long().view(-1)
        )


class MSELossWrapper:
    def __init__(self, num_labels):
        self.__num_labels = num_labels
        self.__loss_fct = nn.MSELoss()

    def calculate_loss(self, logits, labels):
        return self.__loss_fct(logits.view(-1), labels.view(-1))


LOSS_REGISTRY = {
    "classification": CrossEntropyLossWrapper,
    "span_classification": CrossEntropyLossWrapper,
    "regression": MSELossWrapper,
}

