import os
import nltk
import random
import json
import gzip
import glob
import spacy
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp

from collections import Counter
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# data from and extractive processing steps found here:
# https://github.com/HHousen/TransformerSum/blob/master/doc/extractive/datasets.rst


def get_entity(sen1_summ):
    processed_story = nlp(sen1_summ)
    entities = [(x.text, spacy.explain(x.label_)) for x in processed_story.ents]
    try:
        top_entity = Counter(entities).most_common(1)[0][0][0].lower()
    except:
        top_entity = random.choice(list(processed_story.noun_chunks)).lower_
    return top_entity


def onsentence_process(data):
    for story, summ, domain, source in zip(data['story_text'], data['summary'], data['domains'], data['source']):
        split_summ = nltk.sent_tokenize(summ)
        for sen1_summ in split_summ:
            try:
                entity = get_entity(sen1_summ)
            except:
                continue
            story_text_list.append(entity + " [SUBTOPIC] " + story)
            summary_list.append(sen1_summ)
            domains_list.append(domain)
            news_outlet_list.append(source)
    data = {
        'story_text': story_text_list,
        'summary': summary_list,
        'domains': domains_list,
        'source': news_outlet_list
    }
    data = pd.DataFrame(data)
    return data


def process_json_into_text(lines):
    source = []
    target = []
    for line in lines:
        extractive_idx = (np.array(line['labels']) == 1).nonzero()[0]
        src = [' '.join(s) for s in line['src']]
        trg = [src[i] for i in list(extractive_idx)]
        source.append(' '.join(src))
        target.append(' '.join(trg))
    return source, target


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/cnn-dm_extractive/")
    arg_parser.add_argument("--out_dir", default="/home/ubuntu/Datasets/cnn-dm_extractive/")
    arg_parser.add_argument("--make_file_split", action='store_true', default=False)
    arg_parser.add_argument("--sen1_from_csv", action='store_true', default=False)
    args = arg_parser.parse_args()

    if args.make_file_split:
        summ_file_list = glob.glob(os.path.join(args.data_dir, 'cnn_dm_extractive_compressed_5000') + '/*.json.gz')
        print(summ_file_list)
        data = {
            'train': {'source': [], 'target': []},
            'test': {'source': [], 'target': []},
            'val': {'source': [], 'target': []},
        }
        for file in summ_file_list:
            split_ = file.split('/')[-1].split('.')[0]  # train, test, val
            with gzip.open(file, 'r') as f:
                source_list, target_list = process_json_into_text(json.load(f))
                data[split_]['source'].extend(source_list)
                data[split_]['target'].extend(target_list)
        for split_, in_out_data in data.items():
            for type_, data_ in in_out_data.items():  # source, target
                filename = os.path.join(args.out_dir, split_ + '.' + type_)
                with open(filename, 'w') as fw:
                    fw.write('\n'.join(data_) + '\n')
            print("done")
    elif args.sen1_from_csv:
        files = ['train']
        for file in files:
            story_text_list = []
            summary_list = []
            domains_list = []
            news_outlet_list = []
            data = pd.read_csv(os.path.join(args.data_dir, file + '.csv'))
            p = mp.Pool(processes=8)
            split_data = np.array_split(data, 8)
            pool_results = p.map(onsentence_process, split_data)
            p.close()
            p.join()
            parts = pd.concat(pool_results, axis=0)
            parts.to_csv(os.path.join(os.path.join(args.out_dir, file + '_new.csv')), index=False)
    else:
        files = ['train', 'val', 'test']
        for file in files:
            story_text_path = os.path.join(args.data_dir, file + '.source')
            with open(story_text_path, 'r') as story_file:
                story_text = story_file.readlines()
            summary_text_path = os.path.join(args.data_dir, file + '.target')
            with open(summary_text_path, 'r') as summ_file:
                summary = summ_file.readlines()

            news_outlet = []
            cnn_hits = 0
            for article in story_text:
                if "CNN)" in article or "CNN )" in article:
                    news_outlet.append('cnn')
                    cnn_hits += 1
                else:
                    news_outlet.append('dm')
            print(cnn_hits)
            domains = pd.read_json(os.path.join(args.data_dir, file + '.source.domains'), lines=True)
            domains = [row['test_pred_classes'] for row in domains['domain_classifier'].values.tolist()]
            assert len(story_text) == len(domains)

            data = {
                'story_text': story_text,
                'summary': summary,
                'domains': domains,
                'source': news_outlet
            }
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(os.path.join(args.data_dir, '%s.csv' % file)), index=False)

