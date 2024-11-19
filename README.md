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


## Usage

Detailed usage instructions for both tokenizers are provided in their respective directories:

- [pureBPE](./pureBPE/tokenization_demo.ipynb)
- [SKMT](./SKMT/tokenization_demo.ipynb)

## License

This project is licensed under the Apache-2.0 license. See the [LICENSE](./LICENSE) file for details.

## Contact

For any questions or further information, please contact:
- Dávid Držík, Faculty of Natural Science and Informatics, Constantine the Philosopher University in Nitra: [david.drzik@ukf.sk](mailto:david.drzik@ukf.sk)
- František Forgáč, Faculty of Natural Science and Informatics, Constantine the Philosopher University in Nitra


## Citation

If you find our model or paper useful, please consider citing our work:

### Article:
Držík, D., & Forgac, F. (2024). Slovak morphological tokenizer using the Byte-Pair Encoding algorithm. PeerJ Computer Science, 10, e2465. https://doi.org/10.7717/peerj-cs.2465

### BibTeX Entry:
```bib
@article{drzik2024slovak,
  title={Slovak morphological tokenizer using the Byte-Pair Encoding algorithm},
  author={Držík, Dávid and Forgac, František},
  journal={PeerJ Computer Science},
  volume={10},
  pages={e2465},
  year={2024},
  month={11},
  issn={2376-5992},
  doi={10.7717/peerj-cs.2465}
}
```
