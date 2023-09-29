import os
import spacy
import swifter
import argparse
import pandas as pd

from collections import Counter

nlp = spacy.load('en_core_web_sm')
TOP_K_ENT = 5


def get_ner(data):
    story = data['summary'].strip() + '. ' + data['story_text']
    processed_story = nlp(data['summary'].replace("-", " "))
    entities = [(x.text, spacy.explain(x.label_)) for x in processed_story.ents]
    top_entities = Counter(entities).most_common(TOP_K_ENT)
    answer = " <s> ".join([e[0][0] for e in top_entities])
    question = " <s> ".join([e[0][1] for e in top_entities])

    data['answer'] = answer
    data['question'] = question

    return data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/xsum/")
    arg_parser.add_argument("--out_dir", default="/home/ubuntu/Datasets/xsum_ner/")
    arg_parser.add_argument("--remove_none_from_csv", action='store_true', default=False)
    args = arg_parser.parse_args()

    files = ['train', 'val', 'test']

    if args.remove_none_from_csv:
        for file in files:
            data = pd.read_csv(os.path.join(args.out_dir, file + '.csv'))
            new_data = data.dropna()
            new_data.to_csv(os.path.join(os.path.join(args.out_dir, file + '.csv')), index=False)
    else:
        data = pd.read_csv(os.path.join(args.data_dir, 'val' + '.csv'))
        data = data.swifter.allow_dask_on_strings(enable=True).apply(get_ner, axis=1, result_type='expand')
        data = data[['story_text', 'question', 'answer', 'domains']]
        data = data.dropna()
        data.to_csv(os.path.join(os.path.join(args.out_dir, 'val' + '.csv')), index=False)

        data = pd.read_csv(os.path.join(args.data_dir, 'train' + '.csv'))
        data = pd.concat([data, pd.read_csv(os.path.join(args.data_dir, 'test' + '.csv'))], ignore_index=True)
        data = data.swifter.allow_dask_on_strings(enable=True).apply(get_ner, axis=1, result_type='expand')
        data = data[['story_text', 'question', 'answer', 'domains']]
        data = data.dropna()
        data.to_csv(os.path.join(os.path.join(args.out_dir, 'train' + '.csv')), index=False)
