"""Used to find cnn, dailymail, reuters, or unk tags. Also to put everything into csv."""
import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from scripts.preprocessing.xsum_sum_prepreprocess import get_list_txt


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/cnn-dm_subtopic/")
    arg_parser.add_argument("--out_dir", default="/home/ubuntu/Datasets/cnn-dm_subtopic/")
    arg_parser.add_argument("--find_news_outlet", action='store_true', default=False)
    arg_parser.add_argument("--get_csv", action='store_true', default=False)
    args = arg_parser.parse_args()

    if args.find_news_outlet:
        num = ''  # if splitting files manually
        raw_datasets = load_dataset(
            'cnn_dailymail', '3.0.0'
        )
        raw_datasets['val'] = raw_datasets['validation']
        del raw_datasets['validation']

        splits = ['train', 'test', 'val']
        reference = {}
        for split_ in splits:
            with open(args.data_dir + split_ + num + '.target', "r") as f:
                reference[split_] = f.readlines()

        summ_sentence_keys = {}
        last_idx = 0
        for idx, t in enumerate(raw_datasets[split_]['highlights']):
            this_key = ""
            for s in t.split("\n"):
                if this_key == "":
                    last_idx = idx
                if "." not in s and idx == last_idx:
                    this_key += s + " "
                else:
                    this_key += s
                    summ_sentence_keys[this_key.strip()] = idx
                    this_key = ""
                last_idx = idx

        news_outlet = {}
        not_found_count = 0
        for split_ in splits:
            articles = raw_datasets[split_]['article']
            highlights = raw_datasets[split_]['highlights']
            news_outlet[split_] = []
            print("Processing %s" % split_)
            for trg in tqdm(reference[split_], total=len(reference[split_])):
                try:
                    idx = summ_sentence_keys[trg.strip()]
                except:
                    check = np.array([trg.strip() in t.replace("\n", " ") for t in highlights])
                    if sum(check) == 0:
                        not_found_count += 1
                        news_outlet[split_].append('unk')
                        continue
                    idx = (check).nonzero()[0][0]
                article = articles[idx][:25]
                if "(CNN)" in article:
                    news_outlet[split_].append('cnn')
                elif "(Reuters)" in article:
                    news_outlet[split_].append('reuters')
                else:
                    news_outlet[split_].append('dm')
                prev_trg = trg
        print("%s articles not found." % not_found_count)
        for split_ in splits:
            with open(args.data_dir + split_ + num +'.outlet', "w") as fw:
                fw.write('\n'.join(news_outlet[split_]) + '\n')
            print("done")

    if args.get_csv:
        files = ['val', 'test', 'train']
        for file in files:
            data = {
                'story_text': get_list_txt(os.path.join(args.data_dir, file + '.source')),
                'summary': get_list_txt(os.path.join(args.data_dir, file + '.target')),
                'source': get_list_txt(os.path.join(args.data_dir, file + '.outlet'))
            }
            domains = pd.read_json(os.path.join(args.data_dir, file + '.source.domains'), lines=True)
            data['domains'] = [row['test_pred_classes'] for row in domains['domain_classifier'].values.tolist()]
            assert len(data['domains']) == len(data['story_text']) == len(data['summary']) == len(data['source'])
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(os.path.join(args.data_dir, file + '.csv')), index=False)



