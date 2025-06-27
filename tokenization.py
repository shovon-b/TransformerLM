from collections import defaultdict
from typing import Optional
import regex as re
import os
import pickle

import numpy as np

def pretokenizer(texts:list[str], special_tokens:list[str]|None=None) -> tuple[list[list[bytes]], dict[tuple,int], dict[tuple,list[int]]]:
    """
    Args:
        texts: a string of words        
        special_token (optional): a list of special tokens. When it is given,
        the returned nested list of bytes includes the special tokens bytes.
    Returns:
        A nested list of bytes generated from the split pattern
        A dictionary with keys: the count of an adjacent byte pairs in the corpus
            and val: the pair
        A dictionary of with the key: a byte pair and val: a list of pretoken ids
            in which the byte pair occurs.
    """
    if special_tokens is not None:        
        sp_token_dict = {x : 1 for x in special_tokens}  
    else:
        sp_token_dict = {}
   
    PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    pretokens = []
    d = defaultdict(int) #dict to keep the pair count
    d_id = defaultdict(list) #dict to keep the word id in which a pair appears
    pretoken_id = 0    
    for chunk in texts: 
        if sp_token_dict.get(chunk, -1) == 1:
            pretokens.append([chunk.encode("utf-8", errors="ignore")])
            pretoken_id += 1
        else:
            words = re.findall(PAT, chunk)
            for item in words:
                enc_item = [bytes([i]) for i in  item.encode("utf-8", errors="ignore")]
                pretokens.append(enc_item)         
                
                for b in range(len(enc_item) - 1):   
                    pair = (enc_item[b], enc_item[b + 1])
                    d[pair] = d[pair] + 1 
                    d_id[pair].append(pretoken_id)                
                pretoken_id += 1
    return pretokens, d, d_id


def train_bpe(src:str,
              vocab_size:int, 
              vocab_filename:str, 
              merges_filename:str, 
              special_tokens:list[str]|None = None)-> None:    
    """
    Produces a vocabulary and list of mergers.
    Saves the trianed vocabulary on a pkl file named "vocab_filename.pkl" 
    on the "data" folder.
    Saves the leanred mergers on a pkl file named "merges_filename.pkl" 
    on the "data" folder.
    Args:
        src (strong or a file_path): the source of the text to train the bpe tokenizer on. 
        The file is assumed to be saved in the folder "data"
        vocab_size: size of the vocabulary
        vocab_filename: name of the output file where the vocab will be saved. 
        Enter the name only, the extension will be handled automatically.
        merges_filename: name of the output file where the merges will be saved. 
        Enter the name only, the extension will be handled automatically.
        spcial_tokens: an optional list of special tokens
           
    """
    #initialize the dictionary    
    sp_token_bytes = [s.encode("utf-8") for s in special_tokens] if special_tokens is not None else []
    int_bytes = [bytes([i]) for i in range(256)]
    base_bytes = sp_token_bytes + int_bytes
    base_vocab_len = len(base_bytes)
    vocab = dict(zip(range(base_vocab_len), base_bytes))

    if os.path.exists(src):        
        try:           
            with open(src,'r', encoding="utf-8", errors="ignore") as f:
                corpus = f.read()
        except: 
            raise FileNotFoundError("Error reading the input text.")
    else:
        corpus = src
    
    if special_tokens is not None:        
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        #strip-off the special tokens
        sp_PAT = "|".join([re.escape(token) for token in special_tokens])       
        texts = re.split(sp_PAT, corpus)
    else:
        texts = [corpus]   
    
    pretokens, d, d_id = pretokenizer(texts, special_tokens = None)
    merges = []
    for i in range(vocab_size - base_vocab_len):   
        if not d:
            break
        max_count = 1
        max_store = []
        for key in d:
            if d[key] > max_count:
                max_count = d[key]
                max_store = [key]
            elif d[key] == max_count:
                max_store.append(key)
        m_pair = max(max_store) #pair to be merged
        
        new_entry, pos = m_pair[0] + m_pair[1], base_vocab_len + i
        vocab[pos] = new_entry        
        merges.append(m_pair)
        iter_list = d_id[m_pair]
        del d[m_pair]  
        del d_id[m_pair]
        
        for idx in iter_list:            
            word = pretokens[idx]            
            if len(word) == 1:
                continue
            new_word = []
            new_word_idx = 0
            merge_ids = []            
            j = 0
            
            while j < len(word):
                if j < len(word) - 1 and word[j] == m_pair[0] and word[j+1] == m_pair[1]:
                    new_word.append(new_entry)                    
                    merge_ids.append(new_word_idx)
                    new_word_idx += 1
                    if j > 0: 
                        aff_pair = (word[j - 1], word[j])
                        count = d[aff_pair]
                        if count == 1: 
                            del d[aff_pair]
                            del d_id[aff_pair]
                        elif count > 1:
                            d[aff_pair] = count - 1                            
                    if j < len(word) - 2:
                        aff_pair = (word[j + 1], word[j + 2])  
                        count = d[aff_pair]
                        if count == 1:
                            del d[aff_pair]
                            del d_id[aff_pair]
                        elif count > 1:
                            d[aff_pair] = count - 1                            
                    j += 2                
                else:
                    new_word.append(word[j])                     
                    new_word_idx += 1
                    j += 1 
            pretokens[idx] = new_word
            
            for k in merge_ids:
                if k > 0:
                    new_pair = (new_word[k - 1], new_word[k])
                    d[new_pair] = d[new_pair] + 1
                    d_id[new_pair].append(idx)                    
                if k < len(new_word) - 1: 
                    new_pair = (new_word[k], new_word[k + 1])
                    d[new_pair] = d.get(new_pair, 0) + 1
                    d_id[new_pair].append(idx)

    dir_name = "data"
    vocab_path = os.path.join(dir_name, vocab_filename + ".pkl")
    merges_path = os.path.join(dir_name, merges_filename + ".pkl")
    with open(vocab_path, 'wb') as f:       
        pickle.dump(vocab, f)
        print("vocabulary saved")
        
    with open(merges_path, 'wb') as f:       
        pickle.dump(merges, f)
        print("merges saved")


class Tokenizer():
    """
    Tokenizer class for encoding and decoding text data given a dictionary of
    trained vocabulary and a merger list.
    
    """
    def __init__(self, vocab: dict[int, bytes], 
                 merges:list[tuple[bytes, bytes]], 
                 special_tokens:list|None=None):
        
        self.vocab = vocab        
        self.merges = merges
        self.sp_tokens = special_tokens 
        self.reverse_vocab = {val: key for key, val in self.vocab.items()}
        
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str]|None=None)-> 'Tokenizer':
        """
        Creates an instance of the class.
        Args:
           vocab_filepath: file path (w/extension .pkl) where the vocabulary is saved
           merges_filename: file path (w/extension .pkl) where the merges are saved
           special_token: optional argument of special tokens
        """
        if os.path.exists(vocab_filepath):        
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
                print("vocab file loaded!")
        else:
            raise FileNotFoundError("vocab file not found.")
        if os.path.exists(merges_filepath):        
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
                print("merges file loaded!")
        else:
            raise FileNotFoundError("merges file not found.")
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, src:str, output_filename:str|None = None)-> Optional[list[int]]:
        """
        Given the input_filename (w/.txt entension), produces the bpe encoding
        and saves the data as "output_filename".npt format.
        Args:
            src (string or file_path): data to be encoded .
            output_filename: Optional filename of the encoded file to be produced by bpe. 
            If not given, the function returns the encoded tokens.
        """        
    
        if os.path.exists(src):        
            try:           
                with open(src,'r', encoding="utf-8", errors="ignore") as f:
                    corpus = f.read()
            except: 
                raise FileNotFoundError("Error reading the input text.")
        else:
            corpus = src    

        if self.sp_tokens is None:
            texts = [corpus]
            
        else:
            sorted_sp_tokens = sorted(self.sp_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_sp_tokens] 
            PAT0 = "|".join(escaped)        
            texts = re.split(f"({PAT0})", corpus)            
                            
        pretokens, _, d_id = pretokenizer(texts, self.sp_tokens)            
        for n in range(len(self.merges)):
            m_pair = self.merges[n]
            iter_list = d_id[m_pair]
            for idx in iter_list: 
                word = pretokens[idx]
                if len(word) == 1:
                    continue
                new_word = []
                new_word_idx = 0
                merge_ids = []            
                j = 0
                while j < len(word):
                    if j < len(word) - 1 and  word[j] == m_pair[0] and word[j + 1] == m_pair[1]:
                        new_word.append(m_pair[0] + m_pair[1])                    
                        merge_ids.append(new_word_idx)
                        new_word_idx += 1                                             
                        j += 2                
                    else:
                        new_word.append(word[j])                     
                        new_word_idx += 1
                        j += 1 
                pretokens[idx] = new_word
                
                for k in merge_ids:
                    if k > 0:
                        new_pair = (new_word[k - 1], new_word[k])                     
                        d_id[new_pair].append(idx)                    
                    if k < len(new_word) - 1: 
                        new_pair = (new_word[k], new_word[k + 1])                     
                        d_id[new_pair].append(idx) 
                
        tokens = [] 
        for idx in range(len(pretokens)):
            word = pretokens[idx]
            if len(word) == 1:
                tokens.append(self.reverse_vocab[word[0]])
                continue
            else: 
                for i in range(len(word)):
                    tokens.append(self.reverse_vocab[word[i]])   

        
        if output_filename is not None:
            dir_name = "data"
            output_path = os.path.join(dir_name, output_filename + ".npy")
            np.save(output_path, np.array(tokens))
            print("encoding complete!")
        else:
            return tokens

    def decode(self, token_ids:list[int])-> str:
        """
        Given a list of token_ids produces a decoded string.
        Args:
            token_ids: integer valued token ids
        Returns:
            decoded string
        """
        decoded = []
        for idx in range(len(token_ids)):
           decoded.append(self.vocab[token_ids[idx]])     
        return b"".join(decoded).decode("utf-8", errors="replace")

