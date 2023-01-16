# -*- coding: utf-8 -*-
# 💾⚙️🔮

__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    filename="LOG_training.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)

import json
from simpletransformers.ner import NERModel
import torch

from utils import infer_model_type

VALID_LABELS = [
    "OU",
    "OO",
    ".O",
    "!O",
    ",O",
    ".U",
    "!U",
    ",U",
    ":O",
    ";O",
    ":U",
    "'O",
    "-O",
    "?O",
    "?U",
]


def train_model(
    model_type: str = "bert",
    model_name: str = "bert-base-uncased",
    lazy_loading: bool = True,
    num_epochs: int = 3,
    max_seq_length: int = 512,
    overwrite_output_dir: bool = True,
):
    """
    train_model - train a new NER model

                See https://simpletransformers.ai/docs/ner-model/ for more details

    :param str model_type: the type of model to use, defaults to "bert"
    :param str model_name: the name of the model to use, defaults to "bert-base-uncased"
    :param bool lazy_loading: whether to load the model lazily, defaults to True
    :param int num_epochs: maximum number of epochs to train for, defaults to 3
    :param int max_seq_length: maximum sequence length, defaults to 512
    :param bool overwrite_output_dir: whether to overwrite the output directory, defaults to True
    :return tuple: (steps, train_details)
    """
    # Create a NERModel
    model = NERModel(
        model_type,
        model_name,
        args={
            "overwrite_output_dir": overwrite_output_dir,
            "num_train_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "lazy_loading": lazy_loading,
        },
        use_cuda=torch.cuda.is_available(),
        labels=VALID_LABELS,
    )

    # # Train the model
    steps, tr_details = model.train_model("rpunct_train_set.txt")
    return steps, tr_details


def prepare_data():
    """
    Prepares data from Original text into Connnl formatted datasets ready for training
    In addition constraints label space to only labels we care about
    """
    token_data = load_datasets(
        ["telp_train_1.txt", "telp_train_2.txt", "telp_train_3.txt", "telp_train_4.txt"]
    )
    clean_up_labels(token_data, VALID_LABELS)
    eval_set = token_data[-int(len(token_data) * 0.10) :]
    train_set = token_data[: int(len(token_data) * 0.90)]
    create_text_file(train_set, "rpunct_train_set.txt")
    create_text_file(eval_set, "rpunct_test_set.txt")


def load_datasets(dataset_paths):
    """
    Given a list of data paths returns a single data object containing all data slices
    """
    token_data = []
    for d_set in [dataset_paths]:
        with open(d_set, "r") as fp:
            data_slice = json.load(fp)
        token_data.extend(data_slice)
        del data_slice
    return token_data


def get_label_stats(dataset):
    """
    Generates frequency of different labels in the dataset.
    """
    calcs = {}
    for i in dataset:
        for tok in i:
            if tok[2] not in calcs.keys():
                calcs[tok[2]] = 1
            else:
                calcs[tok[2]] += 1
    print(calcs)
    return calcs


def clean_up_labels(dataset, valid_labels):
    """
    Given a list of Valid labels cleans up the dataset
    by limiting to only the labels available.

    In addition prepares observations for training.
    """
    for ix, i in enumerate(dataset):
        for tok in i:
            tok[0] = ix
            if tok[2] not in valid_labels:
                case = tok[2][-1]
                tok[2] = f"O{case}"
                if len(tok[2]) < 2:
                    tok[2] = "OO"


def create_text_file(dataset, name):
    """
    Create Connl ner format file
    """
    with open(name, "w") as fp:
        for obs in dataset:
            for tok in obs:
                line = tok[1] + " " + tok[2] + "\n"
                fp.write(line)
            fp.write("\n")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Train a new NER model using simpletransformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-type",
        "--model_type",
        type=str,
        default=None,
        help="The type of model to use. Defaults to None and will be inferred from the model name",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="The name of the model to use. Defaults to 'bert-base-uncased'",
    )
    parser.add_argument(
        "--no_lazy_loading",
        action="store_true",
        default=False,
        help="Do not use lazy loading",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Maximum number of epochs to train for. Defaults to 3",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length. Defaults to 512",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        type=bool,
        default=True,
        help="Whether to overwrite the output directory. Defaults to True",
    )
    return parser


def e2e_train(args):
    prepare_data()
    steps, tr_details = train_model(
        model_type=args.model_type or infer_model_type(args.model_name),
        model_name=args.model_name,
        lazy_loading=args.lazy_loading,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        overwrite_output_dir=args.overwrite_output_dir,
    )
    print(f"Steps: {steps}; Train details: {tr_details}")


def run():
    parser = get_parser()
    args = parser.parse_args()
    e2e_train(args)


if __name__ == "__main__":
    print("Training the model.")
    e2e_train()
