# -*- coding: utf-8 -*-
"""
💾⚙️🔮
punctuate.py contains the main class that performs punctuation restoration
"""
__author__ = "Daulet N."
__email__ = "daulet.nurmanbetov@gmail.com"

import logging
import shutil

from langdetect import detect
from simpletransformers.ner import NERModel

from rpunct.utils import get_cuda_status, onnx_availability


class RestorePuncts:
    def __init__(
        self,
        model_type="bert",
        model_name="felflare/bert-restore-punctuation",
        wrds_per_pred=250,
        overlap_wrds=30,
        max_seq_length=512,
        overwrite_output_dir=True,
    ):
        """

        RestorePuncts class is the main class that performs punctuation restoration.

        :param str model_type: the type of model to use, defaults to 'bert'
        :param str model_name: the name of the model to use, defaults to 'felflare/bert-restore-punctuation'
        :param int wrds_per_pred: the number of words to feed to the model at once, defaults to 250
        :param int overlap_wrds: overlap between chunks of text, defaults to 30
        :param int max_seq_length: maximum sequence length for the model, defaults to 512
        """
        self.wrds_per_pred = wrds_per_pred
        self.overlap_wrds = overlap_wrds
        self.model_type = model_type
        self.model_name = model_name

        self.valid_labels = [
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
        self.ner_model_args = {
            "silent": True,
            "max_seq_length": max_seq_length,
            "overwrite_output_dir": overwrite_output_dir,
        }
        self.model = NERModel(
            self.model_type,
            self.model_name,
            labels=self.valid_labels,
            args=self.ner_model_args,
            use_cuda=get_cuda_status(),
        )

    def convert_to_onnx(
        self, output_dir: str = "onnx_outputs", remove_local_files: bool = True
    ):
        """
        convert_to_onnx - convert the model to ONNX format for faster inference

           this requires ONNX Runtime to be installed, see
            https://github.com/onnx/onnx#installation
            https://onnxruntime.ai/docs/install/#python-installs


        :param str output_dir: the output directory to save the ONNX model, defaults to 'onnx_outputs'
        :param bool remove_local_files: remove the local files after conversion, defaults to True
        :raises Exception: if ONNX is not installed
        """
        if not onnx_availability():
            raise Exception(
                "ONNX is not available, please install it first: https://onnxruntime.ai/docs/install/#python-installs"
            )

        logging.info("Converting model to ONNX format")
        self.model.convert_to_onnx(output_dir)
        self.ner_model_args.update({"dynamic_quantize": True})

        # load model using existing model_args and the new dynamic_quantize flag from the previous step
        self.model = NERModel(
            self.model_type,
            output_dir,
            labels=self.valid_labels,
            args=self.ner_model_args,
            use_cuda=get_cuda_status(),
        )

        if remove_local_files:
            logging.info("cleaning up onnx export files")
            shutil.rmtree(output_dir, ignore_errors=True)

        logging.info("ONNX conversion complete")

    def punctuate(self, text: str, lang: str = ""):
        """
        Performs punctuation restoration on arbitrarily large text.
        Detects if input is not English, if non-English was detected terminates predictions.
        Overrride by supplying `lang='en'`

        Args:
            - text (str): Text to punctuate, can be few words to as large as you want.
            - lang (str): Explicit language of input text.
        """
        if not lang and len(text) > 10:
            lang = detect(text)
        if lang != "en":
            raise Exception(
                f"""Non English text detected. Restore Punctuation works only for English.
            If you are certain the input is English, pass argument lang='en' to this function.
            Punctuate received: {text}"""
            )

        # plit up large text into bert digestable chunks
        splits = self.split_on_toks(text, self.wrds_per_pred, self.overlap_wrds)
        # predict slices
        # full_preds_lst contains tuple of labels and logits
        full_preds_lst = [self.predict(i["text"]) for i in splits]
        # extract predictions, and discard logits
        preds_lst = [i[0][0] for i in full_preds_lst]
        # join text slices
        combined_preds = self.combine_results(text, preds_lst)
        # create punctuated prediction
        punct_text = self.punctuate_texts(combined_preds)
        return punct_text

    def predict(self, input_slice):
        """
        Passes the unpunctuated text to the model for punctuation.
        """
        predictions, raw_outputs = self.model.predict([input_slice])
        return predictions, raw_outputs

    @staticmethod
    def split_on_toks(text, length, overlap):
        """
        Splits text into predefined slices of overlapping text with indexes (offsets)
        that tie-back to original text.
        This is done to bypass 512 token limit on transformer models by sequentially
        feeding chunks of < 512 toks.
        Example output:
        [{...}, {"text": "...", 'start_idx': 31354, 'end_idx': 32648}, {...}]
        """
        wrds = text.replace("\n", " ").split(" ")
        resp = []
        lst_chunk_idx = 0
        i = 0

        while True:
            # words in the chunk and the overlapping portion
            wrds_len = wrds[(length * i) : (length * (i + 1))]
            wrds_ovlp = wrds[(length * (i + 1)) : ((length * (i + 1)) + overlap)]
            wrds_split = wrds_len + wrds_ovlp

            # Break loop if no more words
            if not wrds_split:
                break

            wrds_str = " ".join(wrds_split)
            nxt_chunk_start_idx = len(" ".join(wrds_len))
            lst_char_idx = len(" ".join(wrds_split))

            resp_obj = {
                "text": wrds_str,
                "start_idx": lst_chunk_idx,
                "end_idx": lst_char_idx + lst_chunk_idx,
            }

            resp.append(resp_obj)
            lst_chunk_idx += nxt_chunk_start_idx + 1
            i += 1
        logging.info(f"Sliced transcript into {len(resp)} slices.")
        return resp

    @staticmethod
    def combine_results(full_text: str, text_slices):
        """
        Given a full text and predictions of each slice combines predictions into a single text again.
        Performs validataion wether text was combined correctly
        """
        split_full_text = full_text.replace("\n", " ").split(" ")
        split_full_text = [i for i in split_full_text if i]
        split_full_text_len = len(split_full_text)
        output_text = []
        index = 0

        if len(text_slices[-1]) <= 3 and len(text_slices) > 1:
            text_slices = text_slices[:-1]

        for _slice in text_slices:
            slice_wrds = len(_slice)
            for ix, wrd in enumerate(_slice):
                # print(index, "|", str(list(wrd.keys())[0]), "|", split_full_text[index])
                if index == split_full_text_len:
                    break

                if (
                    split_full_text[index] == str(list(wrd.keys())[0])
                    and ix <= slice_wrds - 3
                    and text_slices[-1] != _slice
                ):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
                elif (
                    split_full_text[index] == str(list(wrd.keys())[0])
                    and text_slices[-1] == _slice
                ):
                    index += 1
                    pred_item_tuple = list(wrd.items())[0]
                    output_text.append(pred_item_tuple)
        assert [i[0] for i in output_text] == split_full_text
        return output_text

    @staticmethod
    def punctuate_texts(full_pred: list):
        """
        Given a list of Predictions from the model, applies the predictions to text,
        thus punctuating it.
        """
        punct_resp = ""
        for i in full_pred:
            word, label = i
            if label[-1] == "U":
                punct_wrd = word.capitalize()
            else:
                punct_wrd = word

            if label[0] != "O":
                punct_wrd += label[0]

            punct_resp += punct_wrd + " "
        punct_resp = punct_resp.strip()
        # Append trailing period if doesnt exist.
        if punct_resp[-1].isalnum():
            punct_resp += "."
        return punct_resp


if __name__ == "__main__":
    punct_model = RestorePuncts()
    # read test file
    with open("../tests/sample_text.txt", "r") as fp:
        test_sample = fp.read()
    # predict text and print
    punctuated = punct_model.punctuate(test_sample)
    print(punctuated)
