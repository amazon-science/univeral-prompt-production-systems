import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(args):
    files = ['train']
    for file in files:
        story_text_path = os.path.join(args.data_dir, 'inputs', file + '.source')
        with open(story_text_path, 'r') as story_file:
            story_text = story_file.readlines()
        domains = pd.read_json(os.path.join(args.data_dir, 'inputs', file + '.source.domains'), lines=True)
        domains = [row['test_pred_classes'] for row in domains['domain_classifier'].values.tolist()]
        assert len(story_text) == len(domains)

        summ_file_list = glob.glob(os.path.join(args.data_dir, 'outputs') + '/*.txt')
        indices = [int(f.split("/")[-1].split(".")[0]) for f in summ_file_list]
        summary = []
        for summ_file in summ_file_list:
            with open(summ_file, 'r') as f:
                lines = f.readlines()
                summary.append(' '.join(lines))

        story_text = [story_text[idx] for idx in indices]
        domains = [domains[idx] for idx in indices]


        data = {
            'story_text': story_text,
            'summary': summary,
            'domains': domains
        }
        data = pd.DataFrame(data)
        train, val = train_test_split(data, test_size=0.05)
        train.to_csv(os.path.join(os.path.join(args.data_dir, 'train.csv')), index=False)
        val.to_csv(os.path.join(os.path.join(args.data_dir, 'val.csv')), index=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/xsum_extractive/")
    args = arg_parser.parse_args()

    preprocess(args)


