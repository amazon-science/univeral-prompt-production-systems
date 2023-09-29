import yaml
from enum import Enum, IntEnum
from typing import List
from pathlib import Path
from dataclasses import dataclass

column_name_mapping = {
    'abstractive_summarization': ('story_text', 'summary', 'domains'),
    'extractive_summarization': ('story_text', 'summary', 'domains'),
    'topic_extractive_summarization': ('story_text', 'summary', 'domains', 'source'),
    'abstractive_paraphrase': ('inputs', 'outputs', 'domains'),
    'entity_answering': ('story_text', 'question', 'answer', 'domains'),
    'extractive_answering': ('story_text', 'question', 'answer', 'domains'),
    'topic_abstractive_summarization': ('story_text', 'summary', 'domains', 'source'),
    'translate': ('translation',)
}
source_name_mapping = {
    'launch_xsum': 'bbc',
    'newsqa': 'cnn',
}
tokenizer_lang_codes = {
    "cs": "cs_CZ",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "et": "et_EE",
    "fi": "fi_FI",
    "fr": "fr_XX",
    "it": "it_IT",
    "lt": "lt_LT",
    "nl": "nl_XX",
    "ro": "ro_RO",
    "ru": "ru_RU",
}
language_dict = {
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "it": "Italian",
    "lt": "Lithuanian",
    "nl": "Dutch",
    "ro": "Romanian",
    "ru": "Russian",
}
# only works for Europarl
language_pairs = [
    ('bg', 'cs'), ('bg', 'da'), ('bg', 'de'), ('bg', 'el'), ('bg', 'en'), ('bg', 'es'), ('bg', 'et'),
    ('bg', 'fi'), ('bg', 'fr'), ('bg', 'hu'), ('bg', 'it'), ('bg', 'lt'), ('bg', 'lv'), ('bg', 'nl'),
    ('bg', 'pl'), ('bg', 'pt'), ('bg', 'ro'), ('bg', 'sk'), ('bg', 'sl'), ('bg', 'sv'), ('cs', 'da'),
    ('cs', 'de'), ('cs', 'el'), ('cs', 'en'), ('cs', 'es'), ('cs', 'et'), ('cs', 'fi'), ('cs', 'fr'),
    ('cs', 'hu'), ('cs', 'it'), ('cs', 'lt'), ('cs', 'lv'), ('cs', 'nl'), ('cs', 'pl'), ('cs', 'pt'),
    ('cs', 'ro'), ('cs', 'sk'), ('cs', 'sl'), ('cs', 'sv'), ('da', 'de'), ('da', 'el'), ('da', 'en'),
    ('da', 'es'), ('da', 'et'), ('da', 'fi'), ('da', 'fr'), ('da', 'hu'), ('da', 'it'), ('da', 'lt'),
    ('da', 'lv'), ('da', 'nl'), ('da', 'pl'), ('da', 'pt'), ('da', 'ro'), ('da', 'sk'), ('da', 'sl'),
    ('da', 'sv'), ('de', 'el'), ('de', 'en'), ('de', 'es'), ('de', 'et'), ('de', 'fi'), ('de', 'fr'),
    ('de', 'hu'), ('de', 'it'), ('de', 'lt'), ('de', 'lv'), ('de', 'nl'), ('de', 'pl'), ('de', 'pt'),
    ('de', 'ro'), ('de', 'sk'), ('de', 'sl'), ('de', 'sv'), ('el', 'en'), ('el', 'es'), ('el', 'et'),
    ('el', 'fi'), ('el', 'fr'), ('el', 'hu'), ('el', 'it'), ('el', 'lt'), ('el', 'lv'), ('el', 'nl'),
    ('el', 'pl'), ('el', 'pt'), ('el', 'ro'), ('el', 'sk'), ('el', 'sl'), ('el', 'sv'), ('en', 'es'),
    ('en', 'et'), ('en', 'fi'), ('en', 'fr'), ('en', 'hu'), ('en', 'it'), ('en', 'lt'), ('en', 'lv'),
    ('en', 'nl'), ('en', 'pl'), ('en', 'pt'), ('en', 'ro'), ('en', 'sk'), ('en', 'sl'), ('en', 'sv'),
    ('es', 'et'), ('es', 'fi'), ('es', 'fr'), ('es', 'hu'), ('es', 'it'), ('es', 'lt'), ('es', 'lv'),
    ('es', 'nl'), ('es', 'pl'), ('es', 'pt'), ('es', 'ro'), ('es', 'sk'), ('es', 'sl'), ('es', 'sv'),
    ('et', 'fi'), ('et', 'fr'), ('et', 'hu'), ('et', 'it'), ('et', 'lt'), ('et', 'lv'), ('et', 'nl'),
    ('et', 'pl'), ('et', 'pt'), ('et', 'ro'), ('et', 'sk'), ('et', 'sl'), ('et', 'sv'), ('fi', 'fr'),
    ('fi', 'hu'), ('fi', 'it'), ('fi', 'lt'), ('fi', 'lv'), ('fi', 'nl'), ('fi', 'pl'), ('fi', 'pt'),
    ('fi', 'ro'), ('fi', 'sk'), ('fi', 'sl'), ('fi', 'sv'), ('fr', 'hu'), ('fr', 'it'), ('fr', 'lt'),
    ('fr', 'lv'), ('fr', 'nl'), ('fr', 'pl'), ('fr', 'pt'), ('fr', 'ro'), ('fr', 'sk'), ('fr', 'sl'),
    ('fr', 'sv'), ('hu', 'it'), ('hu', 'lt'), ('hu', 'lv'), ('hu', 'nl'), ('hu', 'pl'), ('hu', 'pt'),
    ('hu', 'ro'), ('hu', 'sk'), ('hu', 'sl'), ('hu', 'sv'), ('it', 'lt'), ('it', 'lv'), ('it', 'nl'),
    ('it', 'pl'), ('it', 'pt'), ('it', 'ro'), ('it', 'sk'), ('it', 'sl'), ('it', 'sv'), ('lt', 'lv'),
    ('lt', 'nl'), ('lt', 'pl'), ('lt', 'pt'), ('lt', 'ro'), ('lt', 'sk'), ('lt', 'sl'), ('lt', 'sv'),
    ('lv', 'nl'), ('lv', 'pl'), ('lv', 'pt'), ('lv', 'ro'), ('lv', 'sk'), ('lv', 'sl'), ('lv', 'sv'),
    ('nl', 'pl'), ('nl', 'pt'), ('nl', 'ro'), ('nl', 'sk'), ('nl', 'sl'), ('nl', 'sv'), ('pl', 'pt'),
    ('pl', 'ro'), ('pl', 'sk'), ('pl', 'sl'), ('pl', 'sv'), ('pt', 'ro'), ('pt', 'sk'), ('pt', 'sl'),
    ('pt', 'sv'), ('ro', 'sk'), ('ro', 'sl'), ('ro', 'sv'), ('sk', 'sl'), ('sk', 'sv'), ('sl', 'sv')
]

scan_actions = [
    "jump", "turn", "walk", "run", "look"
]

class Split(Enum):
    train = "train"
    val = "val"
    test = "test"


class DatasetName(Enum):
    xsum_summarize = "XSUM"
    cnndm_topic_summarize = "CNN-DM"
    cnndm_topic_extractive = "CNN-DM"
    xsum_paraphrase = "XSUM"
    xsum_extractive = "XSUM"
    xsum_ner = "XSUM"
    newsqa = "NEWSQA"
    translation = "EUROPARL"


class TaskName(Enum):
    xsum_summarize = "ABSTRACTIVE_SUMMARIZATION"
    cnndm_topic_summarize = "TOPIC_ABSTRACTIVE_SUMMARIZATION"
    cnndm_topic_extractive = "TOPIC_EXTRACTIVE_SUMMARIZATION"
    xsum_paraphrase = "ABSTRACTIVE_PARAPHRASE"
    xsum_extractive = "EXTRACTIVE_SUMMARIZATION"
    xsum_ner = "ENTITY_ANSWERING"
    newsqa = "EXTRACTIVE_ANSWERING"
    translation = "TRANSLATE"


class TaskId(IntEnum):
    xsum_summarize = 0
    cnndm_topic_summarize = 1
    cnndm_topic_extractive = 2
    xsum_paraphrase = 3
    xsum_extractive = 4
    xsum_ner = 5
    newsqa = 6
    translation = 7


@dataclass(init=False)
class Task:
    uid: TaskId
    dataset_name: DatasetName
    name: str
    data_dir: str
    task_name: str
    extension: str
    load_name: str
    config_name: str
    source_lang: str
    target_lang: str

    def __init__(
            self,
            name: str,
            data_dir: str = None,
            dataset_name: str = None,
            task_name: str = None,
            extension: str = None,
            load_name: str = None,
            config_name: str = None,
            source_lang: str = None,
            target_lang: str = None,
            uid: TaskId = None,
            modes: List[Split] = [Split.train, Split.val, Split.test]
    ):
        name = name.lower()
        if "translation" in name:
            name = "translation"
        self.data_dir = data_dir
        self.uid = uid or TaskId[name.replace("-", "_")]
        self.extension = extension
        self.modes = [Split[mode] for mode in modes]
        self.dataset_name = (
            dataset_name
            if isinstance(dataset_name, DatasetName)
            else DatasetName[name]
        )
        self.task_name = (
            task_name
            if isinstance(task_name, TaskName)
            else TaskName[name]
        )
        self.dataset_columns = column_name_mapping.get(self.task_name.value.lower(), None)
        self.source_name = source_name_mapping.get(self.dataset_name.value.lower(), None)
        self.name = name

        # below is used to download directly from HuggingFace data Hub
        self.load_name = load_name
        self.config_name = config_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.source_tokenizer_code = None
        self.target_tokenizer_code = None
        if name == "translation":
            assert self.source_lang is not None or self.target_lang is not None
            self.task_name.from_lang = language_dict[self.source_lang]
            self.task_name.to_lang = language_dict[self.target_lang]
            if self.source_lang not in tokenizer_lang_codes:
                raise ValueError(f"Could not find tokenizer lang code for '{self.source_lang}'")
            self.source_tokenizer_code = tokenizer_lang_codes[self.source_lang]
            if self.target_lang not in tokenizer_lang_codes:
                raise ValueError(f"Could not find tokenizer lang code for '{self.target_lang}'")
            self.target_tokenizer_code = tokenizer_lang_codes[self.target_lang]
            self.task_description = self.task_name.value + \
                                    " %s to %s" % (language_dict[self.source_lang], language_dict[self.target_lang])
            self.task_dict_name = self.task_name.value + \
                                  " %s to %s" % (self.source_lang, self.target_lang)
        else:
            self.task_description = self.task_name.value
            self.task_dict_name = self.task_name.value


def read_tasks_file(data_args) -> List[Task]:
    def _create_task(name, task_data):
        return Task(name=name, **task_data)

    task_file = Path(data_args.tasks_file_path)
    if not task_file.is_file():
        raise ValueError(f"Could not find tasks file at '{data_args.task_file_path}'")
    with task_file.open(mode="r") as f:
        tasks_data = yaml.load(f, Loader=yaml.Loader)
    return [_create_task(k, v) for k, v in tasks_data.items()]


def find_language_pair(source_lang, target_lang):
    for pair in language_pairs:
        if pair == (source_lang, target_lang):
            return source_lang, target_lang
        elif pair == (target_lang, source_lang):
            return target_lang, source_lang
    raise ValueError(f"Could not find language pair for source '{source_lang}' and target '{target_lang}'")

