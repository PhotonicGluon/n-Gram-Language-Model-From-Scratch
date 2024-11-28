# $n$-Gram Language Model From Scratch

This repository contains Python code that implements a rudimentary $n$-Gram language model from scratch.

## Installation

No additional dependencies are required to use [`ngram.py`](ngram.py). However, to run the example, you will need to install [`tqdm`](https://pypi.org/project/tqdm/).

## Usage

See [the example](example).

In brief, the main code is in [`ngram.py`](ngram.py). The main class of concern is `NGramModel`, which constructor takes in one parameter `n` representing the order of the language model.
- The size of the `NGramModel` instance `model` (i.e., number of $n$-grams stored) can be obtained using `model.size`.
- You can save the model by using `model.save()`, and load an existing model by using `NGramModel.load()`. See the example for more details.
  - The `serialize()` and `deserialize()` methods are provided for convenience, and are not required to be used.
- `model.add_ngram()` adds a new $n$-gram to the model. This takes a tuple of length $n$, representing the $n$-gram to be inserted.
- `model.prune()` prunes the model by removing nodes with relative frequencies less than a threshold. See the example for more details.
- To generate text, use `model.generate_text()`. Provide a starting $k$-gram (where $k \leq n$) and the number of words to generate. You can provide a seed to control generation output. The example has more details.

## License

This repository is released into the public domain under [The Unlicense](LICENSE).
