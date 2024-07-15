import torch
#from gpt2.data import Dataset#, Vocab
from typing import Dict, Any, List, Optional
import numpy as np
import random

class Dataset(object):
    def skip(self, count: int):
        raise NotImplementedError()

    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def where(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def assign(self, where: Dict[str, Any]):
        raise NotImplementedError()

class TokenizedCorpus(Dataset):
    def __init__(self,
                 corpus_path: str,
                 vocab: int, #Vocab,
                 seq_len: int,
                 repeat: bool = True):
        #self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.corpus_path = corpus_path
        self.corpus_fp = np.load(corpus_path)
        #self.corpus_fp = np.random.permutation(self.corpus_fp)
        self.vocab = "" #0#vocab
        self.seq_len = seq_len
        self.repeat = repeat
        

    def skip(self, count: int):
        for _ in range(count):
            if not self.corpus_fp.readline():
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                self.corpus_fp.readline()

    def _fetch_one(self) -> Dict[str, List[int]]: 
        """
        Expl: Read a line in the .txt file (corresponding to a sequence) and return the tokenisation of LENGTH:seq_length
        """
        """
        while True:
            # Read subword-tokenized sequence from corpus.
            line = self.corpus_fp.readline()
            #print(len(line))
            #print(line)
            if not line:
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                continue

            # Use token indices rather than the token names directly.
            indices = [self.vocab[t] for t in line.split()]
            #print(len(indices))
            if len(indices) + 2 > self.seq_len:
                print(len(indices) + 2 , self.seq_len)
                continue

            # Decorate the sequence with additional tokens.
            indices = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
            indices += [self.vocab.pad_idx] * (self.seq_len - len(indices) + 1)
            print("SHAPE", len(indices[:-1]))
            return {'input': indices[:-1], 'output': indices[1:]}
        """
        data = self.corpus_fp
        length = len(data)
        sample_int = random.randint(0, length-2000)
        sample = data[sample_int:sample_int+self.seq_len]
        #print("sample shape", sample.shape)
        #a = {'input': sample[:-1], 'output': sample[1:]}
        #print(a)       
        return {'input': sample[:-1], 'output': sample[1:]}


    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        data: a list with len(data)=batch_size and data[i]: dict={'input': tokenised, 'output': tokenised} #
        then puts all "inputs" together and all "outputs" together
        returns: dict = {"inputs": arr_in: batch_size x seq_length (batch_size in number arrays with inputs of seq_length, tokenised)
                        "outputs": arr_out: batch_size x seq_length (batch_size in number arrays with outputs of seq_length, tokenised)}
        """

        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            #print(data[0])
            # for d in data:
            #     #print(d)
            #     count = count + 1
            #     #print(count)
            #     for k in data[0]:
            #         #print(k, "\n")
            #         #print(d[k])

            data = {k: [d[k] for d in data] for k in data[0]}
            #print(data)
            #print("data shape", len(data))#, data[0])
            #print(len(data["input"]))

        #a =  {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
        #print(len(a))
        #print(a["input"].shape)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def where(self) -> Dict[str, Any]:
        return {'offset': self.corpus_path}# self.corpus_fp.tell()}

    def assign(self, where: Dict[str, Any]):
        self.corpus_fp.seek(where['offset'])
