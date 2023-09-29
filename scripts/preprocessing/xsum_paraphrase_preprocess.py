import os
import glob
import argparse
import random
import pandas as pd
from sklearn.model_selection import train_test_split

FILES = ['train', 'val', 'test']


def get_list_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line)
    return data


def prepare_backtranslation_in(args):
    """Creates input files for backtranslation. Randomly picks consecutive sentences."""
    data = []
    for file in FILES:
        data.extend(get_list_txt(os.path.join(args.data_dir, file + '.source')))

    print(len(data))
    for i, txt in enumerate(data):
        txt = txt.replace("\"", "").split(".")
        if txt[-1] == '\n':
            txt = txt[:-1]

        num_sentences = random.randint(2, 4)
        if num_sentences >= len(txt):
            start = 0
        else:
            start = random.randint(0, len(txt)-num_sentences)
        end = start + num_sentences
        txt = ". ".join(l.strip() for l in txt[start:end]) + "."
        with open(os.path.join(args.para_inputs_dir, str(i) + ".txt"), 'w') as file:
            file.write(txt)


def preprocess(args):
    domains = []
    for file in FILES:
        domains_ = pd.read_json(os.path.join(args.data_dir, file + '.source.domains'), lines=True)
        domains_ = [row['test_pred_classes'] for row in domains_['domain_classifier'].values.tolist()]
        domains.extend(domains_)

    inputs = []
    paraphrase = []
    inp_file_list = glob.glob(os.path.join(args.paraphrase_dir, 'input') + '/*.txt')
    out_file_list = glob.glob(os.path.join(args.paraphrase_dir, 'output') + '/*.txt')
    indices = [int(f.split("/")[-1].split(".")[-2]) for f in inp_file_list]
    assert indices.sort() == [int(f.split("/")[-1].split(".")[-2]) for f in out_file_list].sort()

    for inpf, outf in zip(inp_file_list, out_file_list):
        with open(inpf, 'r') as f:
            lines = f.readlines()
            inputs.append(' '.join(lines))
        with open(outf, 'r') as f:
            lines = f.readlines()
            paraphrase.append(' '.join(lines))

    domains = [domains[idx] for idx in indices]

    data = {
        'inputs': inputs,
        'outputs': paraphrase,
        'domains': domains
    }
    data = pd.DataFrame(data)
    train, val = train_test_split(data, test_size=0.05)
    train.to_csv(os.path.join(args.paraphrase_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(args.paraphrase_dir, 'val.csv'), index=False)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", default="/home/ubuntu/Datasets/xsum/")
    arg_parser.add_argument("--paraphrase_dir", default="/home/ubuntu/Datasets/xsum_paraphrase")
    arg_parser.add_argument("--prep_backtranslation_in", action='store_true')
    arg_parser.add_argument("--preprocess", action='store_true')
    args = arg_parser.parse_args()

    if args.prep_backtranslation_in:
        prepare_backtranslation_in(args)

    if args.preprocess:
        preprocess(args)








