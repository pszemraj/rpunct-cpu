# ✏️ rpunct - Restore Punctuation

## rpunct-cpu

This is a fork of [rpunct](https://github.com/Felflare/rpunct) that has some basic improvements to enable CPU usage. It is not intended to be a replacement for the original repo, but rather a _"quick and dirty"_ way to get the model running on a CPU, as the original repo has not been updated in a while.

### Installation

```bash
# create a new virtual environment (optional)
pip install git+https://github.com/pszemraj/rpunct-cpu.git
```

Usage is currently the same as the original repo (see below).

### Improvements

A list of improvements (_WIP_) to investigate/implement:

- [x] get CPU version working & pip installable from this repo
- [ ] investigate more modern models (e.g. roberta, distilbert, etc. for better performance & speed)
- [ ] optimization/quanitization of model for even better performance & speed (_note: this will only be implemented here if the integration with simpletransformers is straightforward_)

---

<center> ⬇️ <strong> below </strong> is the original README from the repo ⬇️ </center>

---

## Original rpunct README

[![forthebadge](https://forthebadge.com/images/badges/made-with-crayons.svg)]()

This repo contains code for Punctuation restoration.

This package is intended for direct use as a punctuation restoration model for the general English language. Alternatively, you can use this for further fine-tuning on domain-specific texts for punctuation restoration tasks.
It uses HuggingFace's `bert-base-uncased` model weights that have been fine-tuned for Punctuation restoration.

Punctuation restoration works on arbitrarily large text.
And uses GPU if it's available otherwise will default to CPU.

List of punctuations we restore:

- Upper-casing
- Period: **.**
- Exclamation: **!**
- Question Mark: **?**
- Comma:  **,**
- Colon:  **:**
- Semi-colon: **;**
- Apostrophe: **'**
- Dash: **-**

---------------------------

### 🚀 Usage

**Below is a quick way to get up and running with the model.**

1. First, install the package.

```bash
pip install rpunct
```

2. Sample python code.

```python
from rpunct import RestorePuncts
# The default language is 'english'
rpunct = RestorePuncts()
rpunct.punctuate("""in 2018 cornell researchers built a high-powered detector that in combination with an algorithm-driven process called ptychography set a world record
by tripling the resolution of a state-of-the-art electron microscope as successful as it was that approach had a weakness it only worked with ultrathin samples that were
a few atoms thick anything thicker would cause the electrons to scatter in ways that could not be disentangled now a team again led by david muller the samuel b eckert
professor of engineering has bested its own record by a factor of two with an electron microscope pixel array detector empad that incorporates even more sophisticated
3d reconstruction algorithms the resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves""")
# Outputs the following:
# In 2018, Cornell researchers built a high-powered detector that, in combination with an algorithm-driven process called Ptychography, set a world record by tripling the
# resolution of a state-of-the-art electron microscope. As successful as it was, that approach had a weakness. It only worked with ultrathin samples that were a few atoms
# thick. Anything thicker would cause the electrons to scatter in ways that could not be disentangled. Now, a team again led by David Muller, the Samuel B.
# Eckert Professor of Engineering, has bested its own record by a factor of two with an Electron microscope pixel array detector empad that incorporates even more
# sophisticated 3d reconstruction algorithms. The resolution is so fine-tuned the only blurring that remains is the thermal jiggling of the atoms themselves.
```

-----------------------------------------------

### 🎯 Accuracy

Here is the number of product reviews we used for finetuning the model:

| Language | Number of text samples|
| -------- | ----------------- |
| English  | 560,000           |

We found the best convergence around _**3 epochs**_, which is what presented here and available via a download.

-----------------------------------------------
The fine-tuned model obtained the following accuracy on 45,990 held-out text samples:

| Accuracy | Overall F1 | Eval Support |
| -------- | ---------------------- | ------------------- |
| 91%  | 90%                 | 45,990

-----------------------------------------------

### 💻🎯 Further Fine-Tuning

To start fine-tuning or training please look into `training/train.py` file.
Running `python training/train.py` will replicate the results of this model.

-----------------------------------------------

### ☕ Contact

Contact [Daulet Nurmanbetov](daulet.nurmanbetov@gmail.com) for questions, feedback and/or requests for similar models.

-----------------------------------------------
