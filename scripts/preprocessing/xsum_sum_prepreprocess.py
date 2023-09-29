import os
import json
import argparse
import pandas as pd


def get_list_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line)
    return data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/xsum/")
    args = arg_parser.parse_args()

    files = ['train', 'val', 'test']
    for file in files:
        data = {
            'story_text': get_list_txt(os.path.join(args.data_dir, file + '.source')),
            'summary': get_list_txt(os.path.join(args.data_dir, file + '.target')),
        }
        domains = pd.read_json(os.path.join(args.data_dir, file + '.source.domains'), lines=True)
        data['domains'] = [row['test_pred_classes'] for row in domains['domain_classifier'].values.tolist()]
        assert len(data['domains']) == len(data['story_text']) == len(data['summary'])
        data = pd.DataFrame(data)
        data.to_csv(os.path.join(os.path.join(args.data_dir, file + '.csv')), index=False)
