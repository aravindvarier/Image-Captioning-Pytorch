import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import bleu

img_dir = './dataset/Flickr8k_Dataset/'
ann_dir = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_dir = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_dir = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_dir = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'

vocab_file = './vocab.txt'

SEED = 123
torch.manual_seed(SEED)
np.random.seed(SEED)

mode = 'test'

class Flickr8kDataset(Dataset):
    """Flickr8k dataset."""
    
    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_dir (string): Directory with all the tokens
            split_dir (string): Directory with all the file names which belong to a certain split(train/dev/test)
            vocab_file (string): File which has the entire vocabulary of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split_dir = split_dir
        self.SOS = self.EOS = None
        self.word_2_token = None
        self.vocab_size = None
        self.image_file_names, self.captions, self.tokenized_captions= self.tokenizer(self.split_dir, self.ann_dir)
        
        if(transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
#                 transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
    
    def tokenizer(self, split_dir, ann_dir):
        image_file_names = []
        captions = []
        tokenized_captions = []
        
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    image_file_names.append(line.split()[0])
                    captions.append(line.split()[1:])


        vocab = []
        # for caption in captions:
        #     for word in caption:
        #         if word not in vocab:
        #             vocab.append(word)
        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        
        self.vocab_size = len(vocab) + 2 #The +2 is to accomodate for the SOS and EOS
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        
        
        self.word_2_token = dict(zip(vocab, list(range(1, self.vocab_size - 1))))

        for caption in captions:
            temp = []
            for word in caption:
                temp.append(self.word_2_token[word])
            temp.insert(0, self.SOS)
            temp.append(self.EOS)
            tokenized_captions.append(temp)
            
        assert(len(image_file_names) == len(captions))
            
        return image_file_names, captions, tokenized_captions
        

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name, cap_tok, caption = self.image_file_names[idx], self.tokenized_captions[idx], self.captions[idx]
        img_name, instance = img_name.split('#')
        img_name = os.path.join(self.img_dir,
                                img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        cap_tok = torch.tensor(cap_tok)
        sample = {'image': image, 'cap_tok': cap_tok, 'caption': caption}

        

        return sample




def collater(batch):
    '''This functions pads the cpations and makes them equal length
    '''
    
    cap_lens = torch.tensor([len(item['cap_tok']) for item in batch]) #Includes SOS and EOS as part of the length
    caption_list = [item['cap_tok'] for item in batch]
#     padded_captions = pad_sequence(caption_list, padding_value=9631) 
    images = torch.stack([item['image'] for item in batch])

    return images, caption_list, cap_lens


def display_sample(sample):
    image = sample['image']
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    image = inv_normalize(image)
    caption = ' '.join(sample['caption'])
    cap_tok = sample['cap_tok']
    plt.figure()
    plt.imshow(image.permute(1,2,0))
    print("Caption: ", caption)
    print("Tokenized Caption: ", cap_tok)
    plt.show()




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*(list(self.model.children())[:8]))
        self.model.requires_grad_(False)
        
    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x,2,3)
        x = x.permute(2,0,1)
        return x



class AdditiveAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(AdditiveAttention, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # A two layer fully-connected network
        self.attention_network = nn.Sequential(
                                    nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(decoder_hidden_size, 1)
                                 )
        self.beta_network = nn.Sequential(nn.Linear(decoder_hidden_size, 1),
                                         nn.Sigmoid())

        self.softmax = nn.Softmax(dim=0)

    def forward(self, queries, keys, values):
        """The forward pass of the additive attention mechanism.

        Arguments:
            queries: The current decoder hidden state. (batch_size x decoder_hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (seq_len x batch_size x encoder_hidden_size)
            values: The encoder hidden states for each step of the input sequence. (seq_len x batch_size x encoder_hidden_size)

        Returns:
            context: weighted average of the values (batch_size x 1 x encoder_hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (seq_len x batch_size x 1)

            The attention_weights must be a softmax weighting over the seq_len annotations.
        """
        batch_size = keys.shape[1]
        seq_len = keys.shape[0]
        expanded_queries = queries.unsqueeze(0).expand(seq_len, batch_size, self.decoder_hidden_size)
        concat_inputs = torch.cat([expanded_queries, keys], dim=2)
        unnormalized_attention = self.attention_network(concat_inputs)
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.permute(1,2,0), values.transpose(0,1))
        beta = self.beta_network(queries)
        context = context * beta.unsqueeze(1)
        return context, attention_weights


class MLP_init(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(MLP_init, self).__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        self.init_MLP = nn.Sequential(
                            nn.Linear(encoder_hidden_size, decoder_hidden_size),
                            nn.ReLU(),
                            nn.Linear(decoder_hidden_size, decoder_hidden_size)
                        )
        
    def forward(self, h):
        return self.init_MLP(h)




class Decoder(nn.Module):
    '''Decode source sequence embeddings into distributions over targets
    '''

    def __init__(
            self, target_vocab_size, pad_id=-1, word_embedding_size=1024,
            encoder_hidden_state_size=1024, decoder_hidden_state_size=1024, cell_type='lstm', dropout=0.0):
        '''Initialize the decoder
        '''
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.pad_id = pad_id
        self.word_embedding_size = word_embedding_size
        self.encoder_hidden_state_size = encoder_hidden_state_size
        self.decoder_hidden_state_size = decoder_hidden_state_size
        self.cell_type = cell_type
        self.dropout = dropout
        self.embedding = self.cell = None
        self.ff_out = self.attention_net = self.ff_init_h = self.ff_init_c = None
        self.init_submodules()

    def init_submodules(self):
        '''Initialize the parameterized submodules of this network
        '''
        self.embedding = nn.Embedding(self.target_vocab_size, self.word_embedding_size, self.pad_id)
        if self.cell_type == 'rnn':
            self.cell = nn.RNNCell(input_size=self.word_embedding_size + self.encoder_hidden_state_size, hidden_size=self.decoder_hidden_state_size)
        elif self.cell_type == 'gru':
            self.cell = nn.GRUCell(input_size=self.word_embedding_size + self.encoder_hidden_state_size, hidden_size=self.decoder_hidden_state_size)
        else:
            self.cell = nn.LSTMCell(input_size=self.word_embedding_size + self.encoder_hidden_state_size, hidden_size=self.decoder_hidden_state_size)
        
        self.ff_out = nn.Linear(self.word_embedding_size + self.decoder_hidden_state_size + self.encoder_hidden_state_size, self.target_vocab_size)
        
        self.ff_init_h = MLP_init(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size)
        self.ff_init_c = MLP_init(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size)
        
        self.attention_net = AdditiveAttention(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, E_tm1, y_tm1, htilde_tm1, h):
        if htilde_tm1 is None:
            htilde_tm1 = self.get_first_hidden_state(h)
            if self.cell_type == 'lstm': #I don't like the way this part's been handled. Handle this later PLEASE!
                ctilde_tm1 = self.ff_init_c(h.mean(axis=0))
                htilde_tm1 = (htilde_tm1, ctilde_tm1)
            
        if self.cell_type == 'lstm':
            xtilde_t, context, attention_weights = self.get_current_rnn_input(E_tm1, htilde_tm1[0], h)
        else:
            xtilde_t, context, attention_weights = self.get_current_rnn_input(E_tm1, htilde_tm1, h)

        h_t = self.get_current_hidden_state(xtilde_t, htilde_tm1)

        if y_tm1 is None: # Change this 
            y_tm1 = self.embedding(torch.zeros(context.shape[0], device=h.device).long())
        else:
            y_tm1 = self.embedding(torch.argmax(y_tm1, axis=1))

        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(h_t[0], y_tm1, context.squeeze(1))
        else:
            logits_t = self.get_current_logits(h_t, y_tm1, context.squeeze(1))

        return logits_t, h_t, attention_weights

    def get_first_hidden_state(self, h):
        '''Get the initial decoder hidden state, prior to the first input
        '''
        h_avg = h.mean(axis=0)
        htilde_tm1 = self.ff_init_h(h_avg)
        
        return htilde_tm1

    def get_current_rnn_input(self, E_tm1, htilde_tm1, h):
        '''Get the current input the decoder RNN
        '''
        context, attention_weights = self.attention_net(htilde_tm1, h, h)
        xtilde_t = torch.cat((self.embedding(E_tm1),context.squeeze(1)),axis=1)
        return xtilde_t, context, attention_weights

    def get_current_hidden_state(self, xtilde_t, htilde_tm1):
        '''Calculate the decoder's current hidden state
        '''
        htilde_t = self.cell(xtilde_t, htilde_tm1)
        return htilde_t

    def get_current_logits(self, htilde_t, y_tm1, ctx_t):
        '''Calculate an un-normalized log distribution over target words
        Uses the deep output layer as described in the paper
        '''
        logits_t = self.ff_out(self.dropout(torch.cat([htilde_t, y_tm1, ctx_t], axis=1)))
        return logits_t


class EncoderDecoder(nn.Module):
    '''Decode a source transcription into a target transcription
    '''

    def __init__(
            self, encoder_class, decoder_class,
            target_vocab_size, target_sos=-2, target_eos=-1, encoder_hidden_size=512,
            decoder_hidden_size=1024, word_embedding_size=1024, cell_type='lstm', beam_width=4, dropout=0.0):
        '''Initialize the encoder decoder combo
        '''
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.target_sos = target_sos
        self.target_eos = target_eos
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.word_embedding_size = word_embedding_size
        self.cell_type = cell_type
        self.beam_width = beam_width
        self.dropout = dropout
        self.encoder = self.decoder = None
        self.init_submodules(encoder_class, decoder_class)
        
    def init_submodules(self, encoder_class, decoder_class):
        '''Initialize encoder and decoder submodules
        '''
        self.encoder = encoder_class()
        self.decoder = decoder_class(self.target_vocab_size, 
                                    self.target_eos, 
                                    self.word_embedding_size, 
                                    self.encoder_hidden_size, 
                                    self.decoder_hidden_size, 
                                    self.cell_type,
                                    self.dropout)

    def get_target_padding_mask(self, E):
        '''Determine what parts of a target sequence batch are padding

        `E` is right-padded with end-of-sequence symbols. This method
        creates a mask of those symbols, excluding the first in every sequence
        (the first eos symbol should not be excluded in the loss).

        Parameters
        ----------
        E : torch.LongTensor
            A float tensor of shape ``(T - 1, N)``, where ``E[t', n]`` is
            the ``t'``-th token id of a gold-standard transcription for the
            ``n``-th source sequence. *Should* exclude the initial
            start-of-sequence token.

        Returns
        -------
        pad_mask : torch.BoolTensor
            A boolean tensor of shape ``(T - 1, N)``, where ``pad_mask[t, n]``
            is :obj:`True` when ``E[t, n]`` is considered padding.
        '''
        pad_mask = E == self.target_eos  # (T - 1, N)
        pad_mask = pad_mask & torch.cat([pad_mask[:1], pad_mask[:-1]], 0)
        return pad_mask

    def forward(self, images, captions=None, max_T=100, on_max='raise'):
        h = self.encoder(images)  # (L, N, H)
        if self.training:
            return self.get_logits_for_teacher_forcing(h, captions)
        else:
            return self.beam_search(h, max_T, on_max)

    def get_logits_for_teacher_forcing(self, h, captions):
        '''Get un-normed distributions over next tokens via teacher forcing
        '''
        op = []
        h_cur = None
        cur_op = None
        total_attention_weights = []
        for i in range(len(captions)-1):
            cur_ip = captions[i]
            cur_op, h_cur, attention_weights = self.decoder(cur_ip, cur_op, h_cur, h)
            op.append(cur_op)
            total_attention_weights.append(attention_weights)
        return torch.stack(op), torch.stack(total_attention_weights)

    def beam_search(self, h, max_T, on_max):
        # beam search
        assert not self.training
        htilde_tm1 = self.decoder.get_first_hidden_state(h)
        logpb_tm1 = torch.where(
            torch.arange(self.beam_width, device=h.device) > 0,  # K
            torch.full_like(
                htilde_tm1[..., 0].unsqueeze(1), -float('inf')),  # k > 0
            torch.zeros_like(
                htilde_tm1[..., 0].unsqueeze(1)),  # k == 0
        )  # (N, K)
        assert torch.all(logpb_tm1[:, 0] == 0.)
        assert torch.all(logpb_tm1[:, 1:] == -float('inf'))
        b_tm1_1 = torch.full_like(  # (t, N, K)
            logpb_tm1, self.target_sos, dtype=torch.long).unsqueeze(0)
        # We treat each beam within the batch as just another batch when
        # computing logits, then recover the original batch dimension by
        # reshaping
        htilde_tm1 = htilde_tm1.unsqueeze(1).repeat(1, self.beam_width, 1)
        htilde_tm1 = htilde_tm1.flatten(end_dim=1)  # (N * K, 2 * H)
        if self.cell_type == 'lstm':
            ctilde_tm1 = self.decoder.ff_init_c(h.mean(axis=0))
            ctilde_tm1 = ctilde_tm1.unsqueeze(1).repeat(1, self.beam_width, 1)
            ctilde_tm1 = ctilde_tm1.flatten(end_dim=1)
            htilde_tm1 = (htilde_tm1, ctilde_tm1)
        h = h.unsqueeze(2).repeat(1, 1, self.beam_width, 1)
        h = h.flatten(1, 2)  # (S, N * K, 2 * H)
        v_is_eos = torch.arange(self.target_vocab_size, device=h.device)
        v_is_eos = v_is_eos == self.target_eos  # (V,)
        t = 0
        logits_tm1 = None
        while torch.any(b_tm1_1[-1, :, 0] != self.target_eos):
            if t == max_T:
                if on_max == 'raise':
                    raise RuntimeError(
                        f'Beam search has not finished by t={t}. Increase the '
                        f'number of parameters and train longer')
                elif on_max == 'halt':
                    warnings.warn(f'Beam search not finished by t={t}. Halted')
                    break
            finished = (b_tm1_1[-1] == self.target_eos)
            E_tm1 = b_tm1_1[-1].flatten()  # (N * K,)
            logits_t, htilde_t, _ = self.decoder(E_tm1, logits_tm1, htilde_tm1, h)
            logits_tm1 = logits_t
            logits_t = logits_t.view(
                -1, self.beam_width, self.target_vocab_size)  # (N, K, V)
            logpy_t = nn.functional.log_softmax(logits_t, -1)
            # We length-normalize the extensions of the unfinished paths
            if t:
                logpb_tm1 = torch.where(
                    finished, logpb_tm1, logpb_tm1 * (t / (t + 1)))
                logpy_t = logpy_t / (t + 1)
            # For any path that's finished:
            # - v == <eos> gets log prob 0
            # - v != <eos> gets log prob -inf
            logpy_t = logpy_t.masked_fill(
                finished.unsqueeze(-1) & v_is_eos, 0.)
            logpy_t = logpy_t.masked_fill(
                finished.unsqueeze(-1) & (~v_is_eos), -float('inf'))
            if self.cell_type == 'lstm':
                htilde_t = (
                    htilde_t[0].view(
                        -1, self.beam_width, self.decoder_hidden_size),
                    htilde_t[1].view(
                        -1, self.beam_width, self.decoder_hidden_size),
                )
            else:
                htilde_t = htilde_t.view(
                    -1, self.beam_width, self.decoder_hidden_size)
            b_t_0, b_t_1, logpb_t = self.update_beam(
                htilde_t, b_tm1_1, logpb_tm1, logpy_t)
            del logits_t, logpy_t, finished, htilde_t
            if self.cell_type == 'lstm':
                htilde_tm1 = (
                    b_t_0[0].flatten(end_dim=1),
                    b_t_0[1].flatten(end_dim=1)
                )
            else:
                htilde_tm1 = b_t_0.flatten(end_dim=1)  # (N * K, 2 * H)
            logpb_tm1, b_tm1_1 = logpb_t, b_t_1
            t += 1
        return b_tm1_1

    def update_beam(self, htilde_t, b_tm1_1, logpb_tm1, logpy_t):
        '''Update the beam in a beam search for the current time step

        Parameters
        ----------
        htilde_t : torch.FloatTensor
            A float tensor of shape
            ``(N, self.beam_with, self.decoder_hidden_size)`` where
            ``htilde_t[n, k, :]`` is the hidden state vector of the ``k``-th
            path in the beam search for batch element ``n`` for the current
            time step. ``htilde_t[n, k, :]`` was used to calculate
            ``logpy_t[n, k, :]``.
        b_tm1_1 : torch.LongTensor
            A long tensor of shape ``(t, N, self.beam_width)`` where
            ``b_tm1_1[t', n, k]`` is the ``t'``-th target token of the
            ``k``-th path of the search for the ``n``-th element in the batch
            up to the previous time step (including the start-of-sequence).
        logpb_tm1 : torch.FloatTensor
            A float tensor of shape ``(N, self.beam_width)`` where
            ``logpb_tm1[n, k]`` is the log-probability of the ``k``-th path
            of the search for the ``n``-th element in the batch up to the
            previous time step. Log-probabilities are sorted such that
            ``logpb_tm1[n, k] >= logpb_tm1[n, k']`` when ``k <= k'``.
        logpy_t : torch.FloatTensor
            A float tensor of shape
            ``(N, self.beam_width, self.target_vocab_size)`` where
            ``logpy_t[n, k, v]`` is the (normalized) conditional
            log-probability of the word ``v`` extending the ``k``-th path in
            the beam search for batch element ``n``. `logpy_t` has been
            modified to account for finished paths (i.e. if ``(n, k)``
            indexes a finished path,
            ``logpy_t[n, k, v] = 0. if v == self.eos else -inf``)

        Returns
        -------
        b_t_0, b_t_1, logpb_t : torch.FloatTensor, torch.LongTensor
            `b_t_0` is a float tensor of shape ``(N, self.beam_width,
            self.decoder_hidden_size)`` of the hidden states of the
            remaining paths after the update. `b_t_1` is a long tensor of shape
            ``(t + 1, N, self.beam_width)`` which provides the token sequences
            of the remaining paths after the update. `logpb_t` is a float
            tensor of the same shape as `logpb_tm1`, indicating the
            log-probabilities of the remaining paths in the beam after the
            update. Paths within a beam are ordered in decreasing log
            probability:
            ``logpb_t[n, k] >= logpb_t[n, k']`` implies ``k <= k'``

        Notes
        -----
        While ``logpb_tm1[n, k]``, ``htilde_t[n, k]``, and ``b_tm1_1[:, n, k]``
        refer to the same path within a beam and so do ``logpb_t[n, k]``,
        ``b_t_0[n, k]``, and ``b_t_1[:, n, k]``,
        it is not necessarily the case that ``logpb_tm1[n, k]`` extends the
        path ``logpb_t[n, k]`` (nor ``b_t_1[:, n, k]`` the path
        ``b_tm1_1[:, n, k]``). This is because candidate paths are re-ranked in
        the update by log-probability. It may be the case that all extensions
        to ``logpb_tm1[n, k]`` are pruned in the update.

        ``b_t_0`` extracts the hidden states from ``htilde_t`` that remain
        after the update.
        '''
        V = logpy_t.shape[2] #Vocab size
        K = logpy_t.shape[1] #Beam width

        s = logpb_tm1.unsqueeze(-1).expand_as(logpy_t) + logpy_t
        logy_flat = torch.flatten(s, 1, 2)
        top_k_val, top_k_ind = torch.topk(logy_flat, K, dim = 1)
        temp = top_k_ind // V #This tells us which beam that top value  is from
        logpb_t = top_k_val

        temp_ = temp.expand_as(b_tm1_1)
        b_t_1 = torch.cat((torch.gather(b_tm1_1, 2, temp_), (top_k_ind % V).unsqueeze(0)))

        if(self.cell_type == 'lstm'):
            temp_ = temp.unsqueeze(-1).expand_as(htilde_t[0])
            b_t_0 = (torch.gather(htilde_t[0], 1, temp_), torch.gather(htilde_t[1], 1, temp_))
        else:
            temp_ = temp.unsqueeze(-1).expand_as(htilde_t)
            b_t_0 = torch.gather(htilde_t, 1, temp_)

        return b_t_0, b_t_1, logpb_t



def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    
    criterion1 = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    criterion2 = nn.MSELoss(reduction='sum')
    total_loss = 0 
    total_num = 0
    for data in tqdm(dataloader):
        images, captions, cap_lens = data
        captions = pad_sequence(captions, padding_value=model.target_eos)
        images, captions, cap_lens = images.to(device), captions.to(device), cap_lens.to(device)
        optimizer.zero_grad()
        logits, total_attention_weights = model(images, captions) #total_attention_weights -> (L, N, 1)
        total_attention_weights = total_attention_weights.sum(axis=0).squeeze(2).T
        captions = captions[1:]
        mask = model.get_target_padding_mask(captions)
        captions = captions.masked_fill(mask,-1)
        loss1 = criterion1(torch.flatten(logits, 0, 1), torch.flatten(captions))
        loss2 = criterion2(total_attention_weights, torch.ones_like(total_attention_weights))
        loss = loss1 + lamda * loss2
        total_loss += loss.item()
        total_num += len(cap_lens)
        loss.backward()
        optimizer.step()
    return total_loss/total_num




CNN_channels = 512 #DO SOMETHING ABOUT THIS
max_epochs = 100
beam_width = 4
decoder_hidden_size = 1800
word_embedding_size = 512
model_save_path = './model_saves/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lamda = 1.
learning_rate = 0.01
dropout = 0.5
batch_size = 64

if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

train_data = Flickr8kDataset(img_dir, train_dir, ann_dir, vocab_file)
val_data = Flickr8kDataset(img_dir, val_dir, ann_dir, vocab_file)
test_data = eval.TestDataset(img_dir, test_dir, ann_dir, vocab_file)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collater)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collater)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=eval.collater)




encoder_class = Encoder
decoder_class = Decoder
model = EncoderDecoder(encoder_class, decoder_class, train_data.vocab_size, target_sos=train_data.SOS, 
                      target_eos=train_data.EOS, encoder_hidden_size=CNN_channels, 
                       decoder_hidden_size=decoder_hidden_size, 
                       word_embedding_size=word_embedding_size, cell_type='lstm', beam_width=beam_width, dropout=dropout)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

def predict(model, device, image_name):
    vocab = []
    with open(vocab_file, "r") as vocab_f:
        for line in vocab_f:
            vocab.append(line.strip())
    image_path = os.path.join(img_dir, image_name)
    print(eval.get_output_sentence(model, device, image_path, vocab))


if mode == "train":
    best_bleu = 0.
    epoch = 1
    while epoch <= max_epochs:
        model.to(device)
        model.train()
        loss = train_for_epoch(model, train_dataloader, optimizer, device)
        model.eval()
        bleu_score = bleu.compute_average_bleu_over_dataset(
            model, val_dataloader,
            val_data.SOS,
            val_data.EOS,
            device,
        )
        print(f'Epoch {epoch}: loss={loss}, BLEU={bleu_score}')
    #     print(f'Epoch {epoch}: loss={loss}')
    #         if bleu_score < best_bleu:
    #             num_poor += 1
    #         else:
    #             num_poor = 0
    #             best_bleu = bleu_score
        if epoch % 50 == 0:
            model.cpu()
            print('Saving Model on Epoch', epoch)
            torch.save(model.state_dict(), model_save_path + 'LSTMAttention.pt')
            
        epoch += 1
        if epoch > max_epochs:
            print(f'Finished {max_epochs} epochs')
        torch.cuda.empty_cache()
elif mode == "test":
    model.load_state_dict(torch.load(model_save_path + '50.pt'))
    model.to(device)
    model.eval()

    predict(model, device, "10815824_2997e03d76.jpg")
    # eval.print_metrics(model, device, test_data, test_dataloader)
    


