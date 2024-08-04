# Slovak Tokenizers: pureBPE and SKMT

This repository contains implementations of two tokenizers developed specifically for the Slovak language: `pureBPE` and `SKMT`.

## Overview

### pureBPE
The `pureBPE` tokenizer is based on the traditional Byte-Pair Encoding (BPE) algorithm and is trained using the Hugging Face implementation. It is designed to efficiently tokenize Slovak text by merging the most frequent pairs of characters or sequences iteratively.

### SKMT (SlovaK Morphological Tokenizer)
The `SKMT` tokenizer incorporates Slovak morphological information into the BPE training process. By utilizing a dictionary of root morphemes, SKMT ensures that word roots remain intact during tokenization, preserving lexical meaning.

## Key Features

- **pureBPE**:
  - Traditional BPE algorithm
  - Optimized for the Slovak language
  - Trained with Hugging Face tools

- **SKMT**:
  - Morphology-aware tokenization
  - Integrates Slovak root morphemes
  - Enhanced lexical preservation

## Benefits

- Improved handling of out-of-vocabulary words
- Reduced vocabulary size
- Better performance in NLP tasks such as sentiment classification
- Higher root integrity in tokenized outputs

## Installation and Usage

Detailed installation and usage instructions for both tokenizers are provided in their respective directories:

- [pureBPE](./pureBPE)
- [SKMT](./SKMT)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or further information, please contact:
- Dávid Držík, Faculty of Natural Science and Informatics, Constantine the Philosopher University in Nitra: [david.drzik@ukf.sk](mailto:david.drzik@ukf.sk)
- František Forgáč, Faculty of Natural Science and Informatics, Constantine the Philosopher University in Nitra
