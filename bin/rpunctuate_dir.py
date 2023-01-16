"""
rpunctuate_dir.py is a script that corrects the punctuation in a directory of text files.

NOTE: running this script requires the following packages:
    clean-text (pip install clean-text)

usage: rpunctuate_dir.py [-h] [-m MODEL] [-type MODEL_TYPE] [-w WRDS_PER_PRED]
                         [-overlap OVERLAP_WRDS] [--max_seq_length MAX_SEQ_LENGTH]
                         [-n ARCHIVE_NAME] [--no_lowercase_inputs]
                         input_dir

"""
import argparse
import logging
import re
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    filename="LOG_repunct_and_PDF_report.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s %(message)s",
)

import warnings

warnings.filterwarnings(action="ignore", message="Blowfish has been deprecated")
from cleantext import clean
from tqdm.auto import tqdm

from rpunct import RestorePuncts


def remove_duplicate_punc(text: str) -> str:
    """
    Remove duplicate consecutive punctuation marks.
    """
    dedup = re.sub(r"([.,!?])[.,!?]+", r"\1", text)
    return dedup


def clean_text(text: str, lower: bool, **kwargs) -> str:
    """
    Clean the text by lowercasing and removing unwanted characters
    """
    return clean(text, lower=lower, **kwargs)


def create_archive(input_dir, archive_name="repunctuated-data") -> Path:
    """
    create_archive is a function that creates a zip archive of the corrected text files.

    :param str or Path input_dir: the directory containing the corrected text files
    :param str archive_name: the name of the archive, defaults to "repunctuated-data"

    :return Path: the path to the archive
    """
    input_dir = Path(input_dir)
    archive_filename = f"rpunctuated-files_{archive_name}.zip"
    archive_path = input_dir.parent / archive_filename
    with zipfile.ZipFile(archive_path, mode="w") as zip_file:
        for f in input_dir.iterdir():
            zip_file.write(f)
    logging.info(f"archive saved to {archive_path.resolve()}")

    return archive_path


def infer_model_type(model_name_or_path: str or Path) -> str:
    """
    infer_model_type - infer the simpletransformers model type from the model name or path
        https://simpletransformers.ai/docs/ner-specifics/

    :param str model_name_or_path: the name or path of the model
    :return str: the model type
    """
    model_name_or_path = Path(model_name_or_path).name
    model_type = re.search(
        r"(bert|xlnet|roberta|mobilebert|deberta-v2|albert)",
        model_name_or_path,
        re.IGNORECASE,
    )
    if model_type:
        return model_type.group(1)
    else:
        raise ValueError(f"model type not recognized: {model_name_or_path}")


def correct_text(rpunct, input_dir: str or Path, lowercase_inputs=True) -> Path:
    """
    correct_text is a function that corrects the punctuation in a text file.

    :param rpunct: an instance of the RestorePuncts class
    :param str or Path input_dir: the directory containing the text files to be corrected
    :param bool lowercase_inputs: whether to lowercase the text before correcting it, defaults to True
    :return Path: the path to the directory containing the corrected text files
    """
    input_dir = Path(input_dir)
    assert input_dir.is_dir(), f"{input_dir} is not a directory"
    rpunct_out = input_dir / "rpunctuated"
    rpunct_out.mkdir(exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.suffix == ".txt"]
    logging.info(f"found {len(files)} text files")

    for f in tqdm(files, desc="repunctuating text"):
        logging.info(f"\nnow starting:\t{f.name}")
        with open(f, "r", encoding="utf-8", errors="ignore") as fi:
            text = clean_text(fi.read(), lower=lowercase_inputs)

        punctuated_text = rpunct.punctuate(text)  # correct the punctuation
        no_dups_text = remove_duplicate_punc(
            punctuated_text
        )  # remove duplicate punctuation (rpunct may add some without removing the originals)

        _outfile = rpunct_out / f"rpunctuated_{f.name}"
        with open(
            _outfile,
            "w",
            encoding="utf-8",
        ) as fo:
            fo.write(no_dups_text)

    return rpunct_out


def get_parser() -> argparse.ArgumentParser:
    """
    get_parser - a function that returns an argparse parser.

    :return argparse.ArgumentParser: parser
    """
    parser = argparse.ArgumentParser(
        description="Restore punctuation in a directory of text files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="felflare/bert-restore-punctuation",
        help="Name of the model to use",
    )
    parser.add_argument(
        "-type",
        "--model_type",
        default=None,
        help="Type of the model to use (if not specified, will be inferred from the model name)",
    )
    parser.add_argument(
        "-w",
        "--wrds_per_pred",
        default=250,
        type=int,
        help="Number of words per prediction",
    )
    parser.add_argument(
        "-overlap",
        "--overlap_wrds",
        default=30,
        type=int,
        help="Number of words to overlap between predictions",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "-n",
        "--archive_name",
        default="repunctuated-data",
        help="Name of the zip archive",
    )
    parser.add_argument(
        "--no_lowercase_inputs",
        default=False,
        action="store_true",
        help="whether to lowercase text when initially read from the input files",
    )
    parser.add_argument(
        "--onnx",
        default=False,
        action="store_true",
        help="whether to use the onnx version of the model",
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing text files to be repunctuated",
    )
    # if no arguments are given, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    return parser


def main(args):
    """
    main - the main function of the script.
    """

    lowercase_inputs = not args.no_lowercase_inputs

    input_dir = Path(args.input_dir)
    assert (
        input_dir.is_dir() and input_dir.exists()
    ), f"{input_dir.resolve()} is not a directory or does not exist"
    rpunct = RestorePuncts(
        model_name=args.model,
        model_type=args.model_type or infer_model_type(args.model),
        wrds_per_pred=args.wrds_per_pred,
        overlap_wrds=args.overlap_wrds,
        max_seq_length=args.max_seq_length,
    )
    if args.onnx:
        rpunct.convert_to_onnx()

    rpunct_out = correct_text(rpunct, input_dir, lowercase_inputs)
    archive_path = create_archive(input_dir=rpunct_out, archive_name=args.archive_name)

    logging.info(f"finished repunctuating {input_dir.name}")
    logging.info(
        f"saved corrected text to:\t{rpunct_out.resolve()}\narchive saved to:\t{archive_path.resolve()}"
    )


def run():
    """
    run - primary entry point
    """
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run()
