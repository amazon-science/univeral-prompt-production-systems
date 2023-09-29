import os
import json
import argparse
import pandas as pd


def get_range(data):
    answer_range = data['answer_token_ranges']
    try:
        start_idx, end_idx = answer_range.split(':')
    except ValueError:
        start_idx, end_idx = answer_range.split(',')[0].split(':')
    start_idx, end_idx = int(start_idx), int(end_idx)
    answer = ' '.join(data['story_text'].split()[start_idx:end_idx])
    return answer


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/newsqa/")
    args = arg_parser.parse_args()

    files = ['train', 'val', 'test']
    for file in files:
        data = pd.read_csv(os.path.join(args.data_dir, file + '.csv'))
        domains = pd.read_json(os.path.join(args.data_dir, file + '.csv.domains'), lines=True)
        domains = [row['test_pred_classes'] for row in domains['domain_classifier'].values.tolist()]
        data['domains'] = pd.DataFrame(domains)
        data['answer'] = data.apply(get_range, axis=1, result_type='expand')
        data = data[['story_text', 'question', 'answer', 'domains']]
        data.to_csv(os.path.join(os.path.join(args.data_dir, file + '.csv')), index=False)
