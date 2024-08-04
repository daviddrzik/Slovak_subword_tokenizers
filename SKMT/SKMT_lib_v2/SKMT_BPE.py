from collections import Counter, defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer
from pathlib import Path
import json
import pickle
import os
import re
from transformers.tokenization_utils_base import BatchEncoding
import torch

class SKMorfoTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.dictionary = None
        self.roots = None
        self.vocab_MDBSNK = None
        self.important_vocab_MDBSNK = None
        self.vocab = None
        self.merges = None
        self.reverse_vocab = None
        self.load_suplementary_files()

    def load_suplementary_files(self):
        current_dir = os.path.dirname(__file__)  # Adresár, kde sa nachádza tento súbor
        root_file = os.path.join(current_dir, 'word_root_20231210_sorted')
        vocab_file = os.path.join(current_dir, 'slova_MDBSNK')
        important_vocab_file = os.path.join(current_dir, 'dolezite_slova_MDBSNK')
        dictionary_file = os.path.join(current_dir, 'kodovanie.json')
        vocab_json_file = os.path.join(current_dir, 'tokenizers/SKMT_BPE/vocab.json')
        merges_txt_file = os.path.join(current_dir, 'tokenizers/SKMT_BPE/merges.txt')
        
        with open(root_file, 'rb') as f:
            self.roots = pickle.load(f)
        
        with open(vocab_file, 'rb') as f:
            self.vocab_MDBSNK = pickle.load(f)
            
        with open(important_vocab_file, 'rb') as f:
            self.important_vocab_MDBSNK = pickle.load(f)
            self.important_vocab_MDBSNK = set(self.important_vocab_MDBSNK)

        with open(dictionary_file, "r", encoding="utf-8") as f:
            self.dictionary = json.load(f)
            
        try:
            with open(vocab_json_file, "r", encoding="utf-8") as file:
                loaded_vocab = json.load(file)
            self.vocab = {prvok: index for prvok, index in loaded_vocab.items()}
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        except FileNotFoundError:
            print("Súbor s vocab neexistuje.")
            
        try:
            with open(merges_txt_file, "r", encoding="utf-8") as file:
                loaded_merges = [tuple(line.split()) for line in file]
            self.merges = {pair: pair[0]+pair[1] for pair in loaded_merges}
        except FileNotFoundError:
            print("Súbor s merges neexistuje.")

    def decode(self, token):
        for k, v in self.dictionary.items():
            if k in token:
                token = token.replace(k, v)
        return token

    def split_word(self, text):
        """Tu sa rozdeluje slovo na znaky a korene, ak korene existujú pre dané slovo"""
        pattern = re.compile(r'§{([^}]+)}§|([^§{}]+)')

        result = []
        for match in pattern.finditer(text):
            inside_brackets, outside_brackets = match.groups()
            if inside_brackets is not None:
                result.append((inside_brackets, 1))
            if outside_brackets is not None:
                result.append((outside_brackets, 0))

        def replace_letters(string):
            for key, value in self.dictionary.items():
                string = re.sub(re.escape(value), key, string)
            return string

        result = [(replace_letters(s), n) for s, n in result]

        new_list = []
        for text, flag in result:
            if flag == 0:
                new_list.extend((char) for char in text)
            elif flag == 1:
                new_list.append((text))
        return new_list
    
    def valid_word(self, word):
        decoded = self.decode(word)
        if decoded.startswith("Ġ"):
            decoded = decoded[1:]
        if decoded[0].lower() in self.vocab_MDBSNK:
            if decoded in self.vocab_MDBSNK[decoded[0].lower()]:
                return True
        return False
    
    def all_words_spaces(self, word_freqs):
        def is_valid_word(word):
            special_chars = "jžxďqitürpľuknŕemfšřýťhzčäwáécóösyoĺěvôdlňabígú"
            pattern = f"^[a-z{special_chars}]+$"
            return re.search(pattern, word) is not None

        def decode(token):
            for k, v in self.dictionary.items():
                if k in token:
                    token = token.replace(k, v)
            return token

        unified_word_freqs = {}

        for word, freq in word_freqs.items():
            if word[0] == 'Ġ':
                if is_valid_word(decode(word[1:])):
                    if unified_word_freqs.get(word, 0) == 0:
                        pokus = word_freqs.get(word[1:], 0)
                        unified_word_freqs[word] = pokus + freq
                else:
                    unified_word_freqs[word] = freq
            else:
                if is_valid_word(decode(word)):
                    if unified_word_freqs.get("Ġ"+word, 0) == 0:
                        pokus = word_freqs.get("Ġ"+word, 0)
                        unified_word_freqs["Ġ"+word] = pokus + freq
                else:
                    unified_word_freqs[word] = freq

        return unified_word_freqs
    
    def all_words_spaces_tokenize(self, tokenized_text):
        def is_valid_word(word):
            special_chars = "jžxďqitürpľuknŕemfšřýťhzčäwáécóösyoĺěvôdlňabígú"
            pattern = f"^[a-z{special_chars}]+$"
            return re.search(pattern, word) is not None

        def decode(token):
            for k, v in self.dictionary.items():
                if k in token:
                    token = token.replace(k, v)
            return token

        unified_tokenized_text = []

        for word in tokenized_text:
            if word[0] == 'Ġ':
                unified_tokenized_text.append(word)
            else:
                if is_valid_word(decode(word)):
                    unified_tokenized_text.append("Ġ"+word)
                else:
                    unified_tokenized_text.append(word)

        return unified_tokenized_text

    def tokenize_half(self, text):
        
        pre_tokenize_result = self.tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
        pre_tokenized_text = [word for word, offset in pre_tokenize_result]
        pre_tokenized_text = self.all_words_spaces_tokenize(pre_tokenized_text)

        splits = {}

        # Use tqdm to create a progress bar for the loop
        for word in pre_tokenized_text:
            decoded = self.decode(word)
            try:
                if decoded.startswith("Ġ"):
                    decoded = decoded[1:]
                    rooted = self.roots[decoded]
                    splits[word] = ["Ġ"] + self.split_word(rooted)
                else:
                    rooted = roots[decoded]
                    splits[word] = self.split_word(rooted)
            except:
                splits[word] = list(word)

        for pair, merge in self.merges.items():
            for idx, split in splits.items():
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2 :]
                    else:
                        i += 1
                splits[idx] = split
                
        zoznam = []
        for slovo in pre_tokenized_text:
            if slovo in splits:
                zoznam.extend(splits[slovo])
        
        return zoznam
    
    def tokenize_additionally(self, word):
        split = list(word)
        
        for pair, merge in self.merges.items():
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
        return split
        
    
    def tokenize(self, text, max_length=None, return_tensors=None, return_subword=False):
        
        casti = text.lower().split("<mask>", 1)
        
        if len(casti) == 1:
            zoznam = self.tokenize_half(text)
        else:
            zoznam = self.tokenize_half(casti[0].strip()) + ["<mask>"] + self.tokenize_half(casti[1])

        # Upraviť input_ids a attention_mask na základe max_length
        if max_length == None:
            return [prvok if prvok in self.vocab else "<unk>" for prvok in zoznam]
        
        # Ak sa token nenachádza v vocab, tak mu priradíme UNK idčko = 3
        input_ids = []
        for prvok in zoznam:
            if prvok in self.vocab:
                input_ids.append(self.vocab[prvok])
            else:
                try:
                    prvky_add = self.tokenize_additionally(prvok)
                    for prvok_add in prvky_add:
                        if prvok_add in self.vocab:
                            input_ids.append(self.vocab[prvok_add])
                        else:
                            input_ids.append(self.vocab["<unk>"])
                except Exception as e:
                    input_ids.append(self.vocab["<unk>"])
        
        if len(input_ids) >= max_length - 2:
            input_ids = input_ids[:max_length - 2]
            attention_mask = [1] * (max_length - 2)
            input_ids = [self.vocab["<s>"]] + input_ids + [self.vocab["</s>"]]
            attention_mask = [1] + attention_mask + [1]
        else:
            padding_length = max_length - len(input_ids) - 2
            input_ids = [self.vocab["<s>"]] + input_ids + [self.vocab["</s>"]]
            attention_mask = [1] * len(input_ids)
            input_ids += [self.vocab["<pad>"]] * padding_length
            attention_mask += [0] * padding_length
            
        # Zmena tu - Zabalíme výsledné tenzory do zoznamu jedného prvku
        output = {"input_ids": [input_ids], "attention_mask": [attention_mask]}
        if return_tensors == "pt":
            output = {key: torch.tensor(val) for key, val in output.items()}

        if return_subword:
            tokens = [self.reverse_vocab[idx] for idx in input_ids]
            return tokens
        
        return BatchEncoding(output)

    def tokenizeQA(self, text1, text2, max_length=None, return_tensors=None, return_subword=False):
        
        zoznam1 = self.tokenize_half(text1.lower().strip())
        zoznam2 = self.tokenize_half(text2.lower().strip())

        # Ak sa token nenachádza v vocab, tak mu priradíme UNK idčko = 3
        input_ids1 = []
        for prvok in zoznam1:
            if prvok in self.vocab:
                input_ids1.append(self.vocab[prvok])
            else:
                # print(f"Nemáme token pre: {prvok}")
                try:
                    prvky_add = self.tokenize_additionally(prvok)
                    for prvok_add in prvky_add:
                        if prvok_add in self.vocab:
                            input_ids1.append(self.vocab[prvok_add])
                        else:
                            input_ids1.append(self.vocab["<unk>"])
                except Exception as e:
                    print(f"Chyba pri spracovaní prvku {prvok}: {e}")
                    input_ids1.append(self.vocab["<unk>"])
        
        # Ak sa token nenachádza v vocab, tak mu priradíme UNK idčko = 3
        input_ids2 = []
        for prvok in zoznam2:
            if prvok in self.vocab:
                input_ids2.append(self.vocab[prvok])
            else:
                # print(f"Nemáme token pre: {prvok}")
                try:
                    prvky_add = self.tokenize_additionally(prvok)
                    for prvok_add in prvky_add:
                        if prvok_add in self.vocab:
                            input_ids2.append(self.vocab[prvok_add])
                        else:
                            input_ids2.append(self.vocab["<unk>"])
                except Exception as e:
                    print(f"Chyba pri spracovaní prvku {prvok}: {e}")
                    input_ids2.append(self.vocab["<unk>"])
        
        total_length = len(input_ids1) + len(input_ids2)

        if total_length >= max_length - 4:
            excess_length = total_length - (max_length - 4)
            while excess_length > 0:
                if len(input_ids1) >= len(input_ids2):
                    input_ids1 = input_ids1[:-1]
                else:
                    input_ids2 = input_ids2[:-1]
                excess_length -= 1

        input_ids1 = [self.vocab["<s>"]] + input_ids1 + [self.vocab["</s>"]]
        input_ids2 = [self.vocab["</s>"]] + input_ids2 + [self.vocab["</s>"]]
        input_ids = input_ids1 + input_ids2

        
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            attention_mask = [1] * (max_length)
        else:
            padding_length = max_length - len(input_ids)
            attention_mask = [1] * len(input_ids)
            input_ids += [self.vocab["<pad>"]] * padding_length
            attention_mask += [0] * padding_length
               
        # Zmena tu - Zabalíme výsledné tenzory do zoznamu jedného prvku
        output = {"input_ids": [input_ids], "attention_mask": [attention_mask]}
        
        if return_tensors == "pt":
            output = {key: torch.tensor(val) for key, val in output.items()}

        if return_subword:
            tokens = [self.reverse_vocab[idx] for idx in input_ids]
            return tokens
        
        return BatchEncoding(output)
    
    def convert_ids_to_tokens(self, input_id):
        return self.decode(self.reverse_vocab[input_id])
    
    def convert_list_ids_to_tokens(self, input_ids):
        tokens = []
        for input_id in input_ids:
            tokens.append(self.decode(self.reverse_vocab[input_id.item() if isinstance(input_id, torch.Tensor) else input_id]))
        return tokens
    
    def convert_tokens_to_ids(self, token):
        return self.vocab[token]
    
    def convert_list_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        return ids
