'''

Note:

# BATCH_SIZE B
# SEQUENCE_LENGTH L
# VOCAB_SIZE V

'''

import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from dataclasses import dataclass
from tensorflow.data import Dataset  # moved here to avoid slow imports
import numpy as np
import osl_tokenize.layers as tokenize_layers
import matplotlib.pyplot as plt
import pickle
import tqdm

@dataclass
class Config():

    '''
    Config class for the token model

    Parameters
    ----------
    SEQUENCE_LENGTH : int
        Length of sequence
    VOCAB_SIZE : int
        Size of vocabulary
    LEARNING_RATE : float
        Learning rate
    CONV_WIDTH : int
        Convolutional width
    RNN_UNITS : int
        Number of RNN units
    BATCH_SIZE : int
        Batch size
    NUM_STAGES : int
        Number of stages
    NUM_EPOCHS_PER_STAGE : int
        Number of epochs per stage

    '''

    model_name: str = "TOKENIZE"

    def __init__(self,
                SEQUENCE_LENGTH = 200,
                VOCAB_SIZE = 128,
                LEARNING_RATE = 0.0001,
                CONV_WIDTH = 10,
                RNN_UNITS = 128,
                BATCH_SIZE = 128,
                NUM_STAGES = 40,
                NUM_EPOCHS_PER_STAGE = 1,
                ):

        self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.CONV_WIDTH = CONV_WIDTH
        self.RNN_UNITS = RNN_UNITS
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_STAGES = NUM_STAGES
        self.NUM_EPOCHS_PER_STAGE = NUM_EPOCHS_PER_STAGE

class Model:

    def __init__(self, config):

        '''
        Token model

        Parameters
        ----------
        config : str or Config
            Config object or path to directory containing config object and weights of the model.
        
        '''

        self.model = None

        if isinstance(config, Config):
            self.config = config

        # if config is string, load model
        elif isinstance(config, str):
            dirname = config

            print("Loading config from", dirname)

            self.config = self._load_config(dirname)

        print("Building model")
        self.build_model()

        self.vocab = {}
        self.vocab['token_order'] = None
        self.vocab['label_map'] = None
        self.vocab['token_counts'] = None
        self.vocab['token_counts_persubj'] = None

        if isinstance(config, str):
            # load model
            self.load(dirname)

    def build_model(self):
        """Builds a keras model."""
        self.model = self._model_structure()

    def fit(self, data_files):

        '''
        Fits the token model to the data in data_files
        
        Arguments
        ---------
        data_files : str or list of str
            List of paths to data files.
            data_files contain numpy arrays in time x channels format
            
        Returns
        -------
        None
        '''

        config = self.config

        ##########
        # Build dataset
        # Note that token_model concatenates channels over the time dimension.

        print("Building dataset...")

        dataset = self.setup_data(data_files, BATCH_SIZE=config.BATCH_SIZE, shuffle=True)

        for count, data in enumerate(dataset):
            if count==0:
                print(f"batch shape is {data['data'].shape}")  # B, L
        print(f"number of batches is {count+1}")  

        ###########
        # Fit model

        print("Compiling model")
        self.model.compile(optimizers.Adam(self.config.LEARNING_RATE))

        print("Fitting model...")

        NUM_EPOCHS = np.ones([config.NUM_STAGES,], dtype=int)*config.NUM_EPOCHS_PER_STAGE
        TEMPERATURES = np.linspace(1,0, config.NUM_STAGES).astype(np.float32)

        histories = []
        for stage, [num_epochs, temperature] in enumerate(zip(NUM_EPOCHS, TEMPERATURES)):
            print(f"Training stage {stage} of {len(NUM_EPOCHS)-1} with temperature {temperature}")
            
            self.token_weights_inference_layer.temperature.assign([temperature])
            history = self.model.fit(dataset, epochs=num_epochs)
            histories.append(history.history['loss'])

        self.histories = histories

        ###########
        # Refactor vocabulary
            
        self.refactor_vocab(data_files, sort=True, trim=True)

        print(f"VOCAB_SIZE={config.VOCAB_SIZE}")
        REFACTORED_VOCAB_SIZE = len(self.vocab['token_order'])+1  
        print(f"REFACTORED_VOCAB_SIZE={REFACTORED_VOCAB_SIZE}")

    def _model_structure(self):

        config = self.config

        inputs = layers.Input(shape=(config.SEQUENCE_LENGTH, 1), name="data",)
        #BATCH_SIZE = tf.shape(inputs)[0]

        rnn_inference_layer = tf.keras.layers.GRU(
                                        config.RNN_UNITS,
                                        return_sequences=True,
                                        stateful=False,
                                        name='rnn_inference_layer'
                                        )
        rnn_inference_output = rnn_inference_layer(inputs)  # shape (B, L, RNN_UNITS)

        self.token_weights_inference_layer = tokenize_layers.TokenWeightsLayer(config.VOCAB_SIZE, )  
        token_weight, temperature = self.token_weights_inference_layer(rnn_inference_output)  # B, L, V

        self.token_basis_layer = tf.keras.layers.Conv1D(
                                filters=config.VOCAB_SIZE,
                                kernel_size=(config.CONV_WIDTH,),
                                padding='same',
                                activation='linear',
                                strides=1,
                                name = 'token_basis_layer')

        signal = self.token_basis_layer(token_weight)  # shape (B, L, V)
        signal = tf.reduce_sum(signal, axis=2, keepdims=True) # shape (B, L, 1)

        nll_layer = tokenize_layers.NegLogNormalLikelihoodLayer()
        nll_loss = nll_layer([signal, tf.ones([1]), inputs])

        return tf.keras.Model(inputs=inputs, outputs=[nll_loss, signal, token_weight, temperature])

    def setup_data(self, data_files, BATCH_SIZE=128, shuffle=True):
        
        '''
        Sets up data in passed in data_files, so that it is ready for model fitting

        Args
        ----
        data_files : list of str
            List of paths to data files.
        BATCH_SIZE : int    
        shuffle : bool      

        Returns
        -------
        dataset is tf.data.Dataset
        '''
        
        config = self.config

        # if data_files is a string, convert to list
        if isinstance(data_files, str):
            data_files = [data_files]

        # iterate over "sessions"
        session_datasets = []
        for sessdatafile in data_files:
            
            print(f"Loading {sessdatafile}")
            
            # load session data
            sessdata = np.load(sessdatafile) # ntpts x nchans
            
            # concatenate channels
            concat_chans_sessdata = np.reshape(sessdata.T, [-1,]) # (n_chans*ntpts,)

            n_seqs = int(np.floor(concat_chans_sessdata.shape[0] / config.SEQUENCE_LENGTH))

            sessdata_trimmed = np.reshape(concat_chans_sessdata[:n_seqs*config.SEQUENCE_LENGTH], [n_seqs, config.SEQUENCE_LENGTH]) # (n_seqs, seq_len)

            # remove mean and standardize
            sessdata_trimmed = sessdata_trimmed - np.mean(sessdata_trimmed, axis=1, keepdims=True)
            sessdata_trimmed = sessdata_trimmed / np.std(sessdata_trimmed, axis=1, keepdims=True)
            sessdata_trimmed = sessdata_trimmed.astype(np.float32) # ntpts x seq_len

            sessdata_trimmed = np.reshape(sessdata_trimmed, [-1,]) # ntpts*seq_len,

            dataset = Dataset.from_tensor_slices({"data": sessdata_trimmed})
            dataset = dataset.batch(config.SEQUENCE_LENGTH, drop_remainder=True)
            
            session_datasets.append(dataset)    

        full_dataset = session_datasets[0]
        for ds in session_datasets[1:]:
            full_dataset = full_dataset.concatenate(ds)

        if shuffle:
            # Shuffle sequences
            full_dataset = full_dataset.shuffle(3000)

            # Group into mini-batches
            full_dataset = full_dataset.batch(BATCH_SIZE)

            # Shuffle mini-batches
            full_dataset = full_dataset.shuffle(1000)

        else:
            # Group into mini-batches
            full_dataset = full_dataset.batch(BATCH_SIZE)

        return full_dataset.prefetch(-1)
    
    def save(self, dirname):
        '''
        Save model.

        Args
        ----
        dirname : str
            Directory to save model to.
        '''

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        np.save(f"{dirname}/histories.npy", self.histories)

        self._save_config(dirname)

        self.model.save_weights(
            f"{dirname}/weights"
        )

        # save token_model.vocab using pickle
        with open(f"{dirname}/vocab.pkl", 'wb') as f:
            pickle.dump(self.vocab, f)

        print('Model saved to:')
        print(dirname)

    def load(self, dirname):
        '''
        Load model from dirname.

        Args
        ----
        dirname : str
            Directory to load the model from.
        
        Returns
        -------
        None

        '''

        print("Loading model from:")
        print(dirname)

        self.histories = np.load(f"{dirname}/histories.npy")

        # Restore weights
        self.model.load_weights(f"{dirname}/weights").expect_partial()

        # load token_model.vocab using pickle
        with open(f"{dirname}/vocab.pkl", 'rb') as f:
            self.vocab = pickle.load(f)

    def _save_config(self, dirname):

        """Save config to a directory."""
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # save token_model.config using pickle
        with open(f"{dirname}/config.pkl", 'wb') as f:
            pickle.dump(self.config, f)

    def _load_config(self, dirname):

        """Load config from a directory."""

        # load token_model.config using pickle
        with open(f"{dirname}/config.pkl", 'rb') as f:
            self.config = pickle.load(f)

        return self.config
    
    def get_vocab(self, input=None):

        '''
        Returns stimulus response for tokens/vocab for passed in input.
        input should be 'impulse' or 'tophat'
        
        Requires self.refactor_vocab to have been run

        Arguments
        ---------
        input : str
            Input to get stimulus response for. Should be 'impulse' or 'tophat'
        
        Returns
        -------
        tokens_refactored : list of numpy array
            List of numpy arrays of shape (NTPTS,) containing stimulus response for each token
        input : numpy array
            Input used to get stimulus response for tokens
        
        '''

        if self.vocab['token_order'] is None:
            raise ValueError("token_order is None. Run refactor_vocab() first.")

        CONV_WIDTH = self.config.CONV_WIDTH

        if input is None or input=="impulse":
            input = np.zeros([CONV_WIDTH*2,])
            input[CONV_WIDTH] = 1
        elif input=="tophat":
            input = np.zeros([CONV_WIDTH*6,])
            input[CONV_WIDTH:CONV_WIDTH*5] = 1  

        tokens = []
        for vv in range(self.config.VOCAB_SIZE):
            NTPTS = input.shape[0]
            token_weights = np.zeros([1, NTPTS, self.config.VOCAB_SIZE])
            token_weights[0, :, vv] = input  # shape (1, NTPTS, V)
            signal_ir = self.token_basis_layer(token_weights).numpy()  # shape (1, NTPTS, V)
            signal_ir = np.sum(signal_ir, axis=2) # shape (1, NTPTS)   
            signal_ir = signal_ir[0, :] # shape (NTPTS)
            tokens.append(signal_ir)

        # refactored tokens
        tokens_refactored = []
        for vv in range(len(self.vocab['token_order'])):
            tokens_refactored.append(tokens[self.vocab['token_order'][vv]])

        return tokens_refactored, input

    def get_pve(self, data_file):

        '''
        Returns percent variance explained by tokens for passed in data_file

        Arguments
        ---------
        data_file : str
            Path to data file.
            data_file contains numpy arrays in time x channels format
        
        Returns
        -------
        pve : list of float
            Percent variance explained for each batch in data_file
        
        '''

        pve = []

        dataset = self.setup_data(data_file, BATCH_SIZE=128, shuffle=False)

        print("Calculating PVE")
        for data_tmp in tqdm.tqdm(dataset):
            _, fitted_signal, _, _ = self.model(data_tmp)

            # flatten fitted signal
            fitted_signal = np.reshape(fitted_signal, [-1])
            data_tmp = np.reshape(data_tmp['data'], [-1])

            # percent variance explained
            pve.append(100*(1 - np.sum((data_tmp - fitted_signal)**2)/np.sum(data_tmp**2)))

        return pve
    
    def refactor_vocab(self, data_files, sort=True, trim=True):

        '''
        Refactors the tokens/vocabulary based on what is needed for the data contained in 
        data_files.
        1) Loads in (list of) data_files and tokenizes 
        2) If trim is True, removes any tokens from vocab that are not used
        3) If sort is True, reorders tokens by how much they get used

        Changes:
        - self.label_map (numpy array containing label map of size VOCAB_SIZE)
        - self.token_order (numpy array containing token order of size VOCAB_SIZE)
        - self.token_counts (numpy array containing token counts of size VOCAB_SIZE)
        - self.token_counts_persubj (numpy array containing token counts per subject of size NUM_SUBJECTS x VOCAB_SIZE)

        Arguments
        ---------
        data_files : str or list of str
            List of paths to data files.
            data_files contain numpy arrays in time x channels format
        sort : bool
            If True, sort tokens by how much they get used
        trim : bool
            If True, remove tokens that are not used

        Returns
        -------
        None

        '''

        config = self.config

        # if data_files is a string, convert to list
        if isinstance(data_files, str):
            data_files = [data_files]

        print("Refactoring vocab")

        token_counts_persubj = np.zeros([len(data_files), config.VOCAB_SIZE])

        for file_index, datfile in enumerate(data_files):
            print(f"Refactoring {datfile}")
            top_token_tc, _ = self._tokenize_data(datfile)

            # get the number of times each token is used
            for ii in range(config.VOCAB_SIZE):
                token_counts_persubj[file_index, ii] = np.sum(top_token_tc==ii)
        
        token_counts = np.sum(token_counts_persubj, axis=0)

        if sort:
            # get token_counts in order
            token_order = np.argsort(token_counts)[::-1]
        else:
            token_order = np.arange(config.VOCAB_SIZE)

        if trim:
            # remove all token indexes with zero counts
            tmp = np.where(token_counts[token_order]==0)[0]
            if tmp.size:
                num_non_zero_tokens = tmp[0]
                token_order = token_order[:num_non_zero_tokens]

        # trim tokens and put them in order
        token_counts = token_counts[token_order]

        token_counts_persubj = token_counts_persubj[:, token_order]

        # Get label_map from token_order
        # Example:
        # If
        # tokens = [A,B,C,D,E,F]
        # token_order = [2,1,4,0], i.e. [C,B,E,A]
        # Then
        # label_map = [4,2,1,0,3,0]
        # i.e. D and F map to 0 indicating they are not used,
        # A maps to 4, B maps to 2, C maps to 1, E maps to 3

        label_map = np.zeros([self.config.VOCAB_SIZE,], dtype=int)
        label_map[token_order] = np.arange(len(token_order))+1

        self.vocab['token_order'] = token_order
        self.vocab['label_map'] = label_map
        self.vocab['token_counts'] = token_counts
        self.vocab['token_counts_persubj'] = token_counts_persubj

    def tokenize_data(self, data_files, data_dir, force=False, random_tokens=False):

        '''
        Tokenize the data in passed in data_file or data_files
        Requires self.fit() and self.refactor_vocab() to have been run first
        
        Arguments
        ---------
        data_files : str or list of str
            List of paths to data files.
            data_files contain numpy arrays in time x channels format
        data_dir : str
            Directory to save tokenized data files to.
        force : bool
            If True, force tokenization even if tokenized data files already exist
        
        Returns
        -------
        tokenized_data_files : str or list of str
            List of paths to tokenized data files.
        '''
        
        if self.vocab['label_map'] is None:
            raise ValueError("label_map is None. Run refactor_vocab() first.")

        # if data_files is a string, convert to list
        string_in = False
        if isinstance(data_files, str):
            data_files = [data_files]
            string_in = True

        skip_first_time = True
        tokenized_data_files = []
        for ss, datfile in enumerate(data_files):

            # get fname datfile
            _, fname = os.path.split(datfile)
            fname, _ = os.path.splitext(fname)
            tokenized_data_file = os.path.join(data_dir, fname + '_tokenized_data.npy')   

            if os.path.exists(tokenized_data_file) and not force and not random_tokens:
                if skip_first_time:
                    print(f"Tokenized data {tokenized_data_file} already exists. Skipping.")
                    skip_first_time = False

                tokenized_data_files.append(tokenized_data_file)
            else:
                
                tokenized_data, tmp = self._tokenize_data(datfile) # channels x time format
 
                # Map from tokenized data labels output from token_model.model to refactored tokenized data labels
                # (where, e.g. tokens are reordered and tokens that are not in the vocab are mapped to 0 (i.e. UNKNOWN)
                #
                # Example:
                #
                # In tokenized_data:
                # token_list = ['A','B','C','D','E']
                # i.e.
                # Token:     A B C D E
                # Label:     1 2 3 4 5
                #
                # If 
                # label_map = [0,2,1,3,0] 
                # i.e.
                # then A and E will be cut from the vocab and
                # in refactored_tokenized_data:
                # Token:     C B D UNKNOWN
                # Label:     1 2 3    0
                # where UNKNOWN is NOT(B,C,D)
                #
                # So if sequence is [C,A,C,D,D] then
                # tokenized_data = [2,0,2,3,3]
                # refactored_tokenized_data_sub is [1,0,1,3,3]
                # i.e. refactored sequence is [C,U,C,D,D] where U is UNKNOWN

                # map from (ntpts x nchans) tokenized_data to (ntpts x nchans) refactored_tokenized_data
                refactored_tokenized_data_sub = np.zeros([tokenized_data.shape[0], tokenized_data.shape[1]], dtype=int)
                for cc in range(tokenized_data.shape[1]):
                    refactored_tokenized_data_sub[:, cc] = [self.vocab['label_map'][x] for x in tokenized_data[:, cc]]

                if random_tokens:
                    # randomize tokens
                    REFACTORED_VOCAB_SIZE = len(self.vocab['token_order'])+1  
                    refactored_tokenized_data_sub = np.random.randint(1, REFACTORED_VOCAB_SIZE, size=refactored_tokenized_data_sub.shape)  
                    
                # save refactored_tokenized_data_sub
                np.save(tokenized_data_file, refactored_tokenized_data_sub) 
                tokenized_data_files.append(tokenized_data_file)

        if string_in:
            tokenized_data_files = tokenized_data_files[0]

        return tokenized_data_files

    def _tokenize_data(self, data_file):

        # load data
        dataset = self.setup_data(data_file, BATCH_SIZE=128, shuffle=False)
        dat_shape = np.load(data_file).shape
        print("dat_shape", dat_shape)
        n_chans = dat_shape[1]

        token_weights_all = []

        print("Tokenizing data")
        for data_tmp in tqdm.tqdm(dataset):
            _, _, token_weights, _ = self.model(data_tmp)

            token_weights_all.append(token_weights) # batch_size, seq_len, vocab_size

        # concatenate ragged arrays in list over their zeroth dimension
        token_weights_all = np.concatenate(token_weights_all, axis=0) # nbatch*batch_size, seq_len, vocab_size
        token_weights_all = np.reshape(token_weights_all, [-1, self.config.VOCAB_SIZE]) # nbatch*batch_size*seq_len, vocab_size

        # add back in trimmed part (zeroed) to ensure output is same size as dat
        token_weights_all_untrimmed = np.zeros([dat_shape[0]*dat_shape[1], self.config.VOCAB_SIZE]) # (ntpts*nchans, vocab_size)
        token_weights_all_untrimmed[:token_weights_all.shape[0], :] = token_weights_all  # (ntpts*nchans, vocab_size)
        
        # put into (channels x time x vocab_size) format
        token_weights_all = np.transpose(np.reshape(token_weights_all_untrimmed, [n_chans, -1, self.config.VOCAB_SIZE]), [1, 0, 2]) # (n_tpts, n_chans, vocab_size)

        # get the best token for each time point
        top_token_all = np.zeros([token_weights_all.shape[0], token_weights_all.shape[1]], dtype=int)
        for cc in range(token_weights_all.shape[1]):
            top_token_all[:, cc] = np.argmax(token_weights_all[:, cc, :], axis=1)

        return top_token_all, token_weights_all
    
    def reconstruct_data(self, tokenized_data):

        # Reconstructs data from passed in tokens
        #
        # Arguments
        # ---------
        # 
        # refactored_tokenized_data - should be ntpts x nchans, or string of file path to tokenized data
        #
        #
        # Returns
        # -------
        #
        # reconstructed_data - ntpts x nchans

        if self.vocab['label_map'] is None:
            raise ValueError("label_map is None. Run refactor_vocab() first.")

        if isinstance(tokenized_data, str):
            if not os.path.exists(tokenized_data):
                raise ValueError(f"tokenized_data {tokenized_data} does not exist.")
            
            tokenized_data = np.load(tokenized_data)

        # map from (ntpts x nchans) refactored_tokenized_data back to (ntpts x nchans) tokenized_data
        tokenized_data_tmp = np.zeros([tokenized_data.shape[0], tokenized_data.shape[1]], dtype=int)
        for cc in range(tokenized_data.shape[1]):
            tokenized_data_tmp[:, cc] = [np.where(np.array(self.vocab['label_map'])==x)[0][0] for x in tokenized_data[:, cc]]

        return self._reconstruct_data(tokenized_data_tmp)

    def _reconstruct_data(self, tokenized_data):

        # convert one-hot vector to vector
        token_weight = np.zeros([tokenized_data.shape[0], tokenized_data.shape[1], self.config.VOCAB_SIZE])
        token_weight[np.arange(tokenized_data.shape[0])[:, None], np.arange(tokenized_data.shape[1]), tokenized_data] = 1

        reconstructed_data = np.zeros([tokenized_data.shape[0], tokenized_data.shape[1]])
        nchans = token_weight.shape[1]
        
        print(f"Reconstructing {nchans} channels")
        for chan in tqdm.tqdm(range(nchans)):
            token_weight_chan = np.reshape(token_weight[:, chan, :], [1, -1, self.config.VOCAB_SIZE])  # shape (tpts, 200, V)
            reconstructed_data_chan = self.token_basis_layer(token_weight_chan)  # shape (tpts, 1, V)
            reconstructed_data_chan = np.sum(reconstructed_data_chan, axis=2) # shape (tpts, 1)
            reconstructed_data[:, chan] = np.squeeze(reconstructed_data_chan)

        return reconstructed_data

def plot_history(histories, plot_dir=None):

    '''
    Plots training history

    Arguments
    ---------
    histories : list of list of float
        List of histories. Each history is a list of floats.
    plot_dir : str
        Directory to save plot to. If None, plot is not saved.
    
    Returns
    -------
    None

    '''

    # create plot dir if it doesn't exist
    if plot_dir is not None:
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)   

    hists = None
    for hist in histories:
        if hists is None:
            hists = np.array(hist)
        else:
            hists = np.concatenate([hists, np.array(hist)])

    plt.figure()
    plt.plot(hists)
    plt.ylabel('loss')
    plt.xlabel('epochs')

    if plot_dir is not None:
        plt.savefig(os.path.join(plot_dir, 'history.png'))
