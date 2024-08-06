import argparse
import torch.nn as nn
#rom gpt2.data import Vocab, Tokenizer
from gpt2.modeling import Transformer
from gpt2.generation import GenerationSpec, GenerateConfig, Generator
from typing import List
import numpy as np
import random 

class GPT2GenerationSpec(GenerationSpec):
    def __init__(self, vocab_path: str, seq_len: int, layers: int, heads: int,
                 dims: int, rate: int):
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate

    def initialize(self):
        self.vocab = ""#Vocab(vocab_path=self.vocab_path)
        self.tokenizer = ""#Tokenizer(vocab=self.vocab)

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=100,#self.vocab.pad_idx,
                           words=51, #len(self.vocab), 
                           seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=0, bidirectional=False)

    def encode_context(self, context: str) -> List[int]:
        tokens = [self.vocab[t] for t in self.tokenizer.encode(context)]
        tokens = [self.vocab.bos_idx] + tokens

        return tokens

    def decode_tokens(self, tokens: List[int]) -> str:
        if self.vocab.eos_idx in tokens:
            tokens = tokens[:tokens.index(self.vocab.eos_idx)+1]
        return self.tokenizer.decode([self.vocab[t] for t in tokens])


def generate_sentence_with_gpt2_model(args: argparse.Namespace):
    spec = GPT2GenerationSpec(
        vocab_path=args.vocab_path, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate)
    config = GenerateConfig(
        seq_len=args.seq_len, nucleus_prob=args.nucleus_prob,
        use_gpu=args.use_gpu)

    generator = Generator(spec, config)
    generator.initialize(from_model=args.model_path)
    no_points = 100 #2 #int(args.seq_len/2)
    no_generated = 100
    print("Take {} random timepoints from data in file: ".format(no_points), args.data_path)
    data = np.load(args.data_path)
    length = len(data)
    count = 0
    data_generated = np.zeros((no_points + args.num_generated_samples * no_points))
    data_generated = np.zeros((no_points + args.num_generated_samples * no_generated))
    print("len data gen", len(data_generated))
    # take x random points
    sample_int = random.randint(0, length-2000)
    points = data[sample_int:sample_int+no_points]
    data_generated[0:no_points] = points
    input = ""
    for k in range(no_points):
            input = input + str(points[k]) + " "

    while count<args.num_generated_samples:
        count = count + 1
        
        #print(input)
             
        print(count)
        #print(input)
        tokens = generator.generate(input)
        #print(len(tokens))
        #print("added no.", no_points+(count-1)*no_generated, no_points+count*no_generated)
        #print("added", -no_points,-no_points+no_generated)
        if -no_points+no_generated<0:
            points = tokens[-no_points:-no_points+no_generated]
            data_generated[no_points+(count-1)*no_generated:no_points+count*no_generated] = tokens[-no_points:-no_points+no_generated]
        else:
            points = tokens[-no_points:]
            data_generated[no_points+(count-1)*no_generated:no_points+count*no_generated] = tokens[-no_points:]

        input = ""
        #points = tokens[-no_generated:]
        #points = tokens[-args.seq_len + no_generated : -no_points+no_generated]
        
        #print(len(points)) ## always predict no_generated symbols based on the no_points
        for k in range(len(points)):
            input = input + str(points[k]) + " "
        
        #print("add ", points)
        #                           # and use them as input for the next generation.
        #print("exrta generated ", tokens[no_points:])
        #print("all data", data_generated)

    ## Save generated data.
    print("save data")
    np.save('generated_data_recursively.npy', data_generated)

    ## tokens -> signal: Reverse tokeniser. TODO.
    #signal = token_decoder(data_generated)



def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'generate', help='generate sentences with GPT-2 model')

    parser.add_argument('--vocab_path', required=True,
                        help='vocabulary file path')
    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')
    parser.add_argument('--data_path', required=True,
                        help='data file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=6, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=8, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')

    group = parser.add_argument_group('Generation options')
    group.add_argument('--nucleus_prob', default=0.85, type=float,
                       help='probability threshold for nucleus sampling')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')
    group.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
    group.add_argument('--num_generated_samples', default=100, type=int,
                       help='number of gerated samples')

    parser.set_defaults(func=generate_sentence_with_gpt2_model)
