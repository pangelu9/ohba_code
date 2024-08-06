## Create virtual environment

```
conda env create -f environment.yml
conda activate osld
```

## Simulate data

```
python main_generate.py --type hmm --validate --tde --hmm_epochs 50 --num_tde 5
```
--validate: if the argument is passed a HMM is fitted on the simulated data. The fitting and statistics of the HMM are plotted and the Dice coefficient is given.

## Tokenise data using osl-tokeniser

```
cd osl-tokenize/examples
python run_tokenizer.py --do_train --dataset_name load_dataset --load_dataset_dir ./dev/results/raw_data/load_dataset_snr5_cha1_sub1_gro1_mod1/meg_data.npy 
```

Inverser proceess, de-tokenise data:
```
python run_tokenizer.py  --do_reconstruct --dataset_name load_dataset --load_dataset_dir ./dev/results/raw_data/load_dataset_snr5_cha1_sub1_gro1_mod1/meg_data.npy --tokenised_dir ./dev/results/osl-tokenize/load_dataset_snr5_cha1_sub1_gro1_mod1
```

## GPT2

### Train model

```
nohup python -m gpt2 train --train_corpus train_data_SNR3.npy --eval_corpus val_data_SNR3.npy --save_checkpoint_path ckpt-gpt2.pth --save_model_path gpt2-model.pth --batch_train 64 --batch_eval 128 --seq_len 200 --total_steps 50000 --eval_steps 500 --save_steps 2000 --base_lr 1e-4
```

### Visualise results
```
python -m gpt2 visualize --model_path gpt2-model.pth --interactive --val_incl --eval_every 500
```

--val_incl: includes the validation curves

--eval_every: every how many steps validation was performed during training


### Generate data
```
 python -m gpt2 generate --model_path gpt2-model.pth --seq_len 200 --nucleus_prob 0.8 --data_path val_data_SNR3.npy --num_generated_samples 1000
```
--data_path: a 100 token part of the validation signal is chosen randomly to be fed as prompt to the model to generate data.

## Analysis on the GPT2-generated data
```
cd ../../osl-tokenize/examples
python run_tokenizer.py --do_reconstruct --dataset_name load_dataset --load_dataset_dir x --tokenised_dir ./dev/results/osl-tokenize/load_dataset_snr5_cha1_sub1_gro1_mod1
```
