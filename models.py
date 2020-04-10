import torch
import torch.nn as nn
import torchvision.models as models
import warnings

class Encoder(nn.Module):
    """
    Encoder.
    """
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
    
    # def __init__(self, encoded_image_size=14):
    #     super(Encoder, self).__init__()
    #     self.enc_image_size = encoded_image_size

    #     resnet = models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

    #     # Remove linear and pool layers (since we're not doing classification)
    #     modules = list(resnet.children())[:-2]
    #     self.resnet = nn.Sequential(*modules)

    #     # Resize image to fixed size to allow input images of variable size
    #     self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    #     self.fine_tune()

    # def forward(self, images):
    #     """
    #     Forward propagation.
    #     :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
    #     :return: encoded images
    #     """
    #     out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
    #     # out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
    #     out = torch.flatten(out,2,3)  #(batch_size, 2048, encoded_image_size * encoded_image_size)
    #     out = out.permute(2, 0, 1)  # (encoded_image_size * encoded_image_size, batch_size, 2048)
    #     return out

    # def fine_tune(self, fine_tune=False):
    #     """
    #     Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
    #     :param fine_tune: Allow?
    #     """
    #     for p in self.resnet.parameters():
    #         p.requires_grad = False
    #     # If fine-tuning, only fine-tune convolutional blocks 2 through 4
    #     for c in list(self.resnet.children())[5:]:
    #         for p in c.parameters():
    #             p.requires_grad = fine_tune



class AdditiveAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, attention_dim):
        super(AdditiveAttention, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_dim = attention_dim

        self.beta_network = nn.Sequential(nn.Linear(decoder_hidden_size, 1),
                                         nn.Sigmoid())

        self.softmax = nn.Softmax(dim=0)

        self.decoder_att_net = nn.Linear(decoder_hidden_size, attention_dim)
        self.encoder_att_net = nn.Linear(encoder_hidden_size, attention_dim)
        self.full_att_net = nn.Sequential(nn.Linear(attention_dim, 1),
                                            nn.ReLU())


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

        Note: seq_len here refers to H*W (which by default is 14*14 = 196)
        """
        batch_size = keys.shape[1]
        seq_len = keys.shape[0]
        expanded_queries = queries.unsqueeze(0).expand(seq_len, batch_size, self.decoder_hidden_size)
        att1 = self.decoder_att_net(expanded_queries)
        att2 = self.encoder_att_net(keys)
        unnormalized_attention = self.full_att_net(att1 + att2)
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
            encoder_hidden_state_size=1024, decoder_hidden_state_size=1024, attention_dim=512, cell_type='lstm', dropout=0.0):
        '''Initialize the decoder
        '''
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.pad_id = pad_id
        self.word_embedding_size = word_embedding_size
        self.encoder_hidden_state_size = encoder_hidden_state_size
        self.decoder_hidden_state_size = decoder_hidden_state_size
        self.attention_dim = attention_dim
        self.cell_type = cell_type
        self.dropout = dropout
        self.embedding = self.cell = None
        self.ff_out = self.attention_net = self.ff_init_h = self.ff_init_c = None
        self.init_submodules()
        self.init_weights()

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
        
        self.ff_out = nn.Linear(self.word_embedding_size , self.target_vocab_size)
        
        self.ff_init_h = MLP_init(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size)
        self.ff_init_c = MLP_init(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size)
        
        self.attention_net = AdditiveAttention(encoder_hidden_size=self.encoder_hidden_state_size, decoder_hidden_size=self.decoder_hidden_state_size,
                                                attention_dim=self.attention_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        self.output_linear_1 = nn.Linear(self.decoder_hidden_state_size, self.word_embedding_size)
        self.output_linear_2 = nn.Linear(self.encoder_hidden_state_size, self.word_embedding_size)

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        # self.embedding.weight.data.uniform_(-0.1, 0.1)
        # nn.init.xavier_uniform_(self.embedding.weight)
        # self.ff_out.bias.data.fill_(0)
        # self.ff_out.weight.data.uniform_(-0.1, 0.1)
        # nn.init.xavier_uniform_(self.ff_out.weight)
        # nn.init.xavier_uniform_(self.output_linear_1.weight)
        # nn.init.xavier_uniform_(self.output_linear_2.weight)
        # self.output_linear_1.weight.data.uniform_(-0.1, 0.1)
        # self.output_linear_2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, E_tm1, y_tm1, htilde_tm1, h):
        '''
        Inputs:
        h:  Encoder hidden states. #(H*W, batch_size, num_channels)
        htilde_tm1: Previous decoder hidden state. #(batch_size, decoder_hidden_size)
                    Tuple of two of these if cell type is LSTM
        E_tm1: Current input. #(batch_size, )
        y_tm1: Previous output logits. #(batch_size, vocab_size)

        Returns:
        logits_t: Output logits #(batch_size, vocab_size)
        htilde_t: Current decoder hidden state. #(batch_size, decoder_hidden_size)
                    Tuple of two of these if cell type is LSTM
        attention_weights: All the attention weights. #(num_encoder_hidden_states, batch_size, 1)
        '''
        if htilde_tm1 is None:
            htilde_tm1 = self.get_first_hidden_state(h)
            if self.cell_type == 'lstm': #I don't like the way this part's been handled. Handle this later PLEASE!
                ctilde_tm1 = self.ff_init_c(h.mean(axis=0))
                htilde_tm1 = (htilde_tm1, ctilde_tm1)
            
        if self.cell_type == 'lstm':
            xtilde_t, context, attention_weights = self.get_current_rnn_input(E_tm1, htilde_tm1[0], h)
            #context: (batch_size, 1, encoder_hidden_size) just a reminder, encoder_hidden_size is the num of channels
            #xtilde_t: (batch_size, embedding_size + encoder_hidden_size)
            #attention_weights: (num_encoder_hidden_states, batch_size, 1) just a reminder, num_encoder_hidden_states is H*W (default 14*14=196)
        else:
            xtilde_t, context, attention_weights = self.get_current_rnn_input(E_tm1, htilde_tm1, h)

        htilde_t = self.get_current_hidden_state(xtilde_t, htilde_tm1) #Same shape as htilde_tm1

        if y_tm1 is None: # Change this 
            y_tm1 = self.embedding(torch.zeros(context.shape[0], device=h.device).long())
        else:
            y_tm1 = self.embedding(torch.argmax(y_tm1, axis=1)) #(batch_size, embedding_size)

        if self.cell_type == 'lstm':
            logits_t = self.get_current_logits(htilde_t[0], y_tm1, context.squeeze(1))
        else:
            logits_t = self.get_current_logits(htilde_t, y_tm1, context.squeeze(1)) #(batch_size, vocab_size)

        return logits_t, htilde_t, attention_weights

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
        logits_t = self.ff_out(self.dropout(self.output_linear_1(htilde_t) + y_tm1 + self.output_linear_2(ctx_t)))
        return logits_t

class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN
        # ------------
        batch_size = queries.shape[0]
        q = self.Q(queries.view(batch_size, -1, queries.shape[-1]))
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = k@q.transpose(2,1)*self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        context = attention_weights.transpose(2,1)@v
        return context, attention_weights
        

class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = torch.tensor(-1e7)

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(torch.tensor(self.hidden_size, dtype= torch.float))

    def forward(self, queries, keys, values):
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        # ------------
        # FILL THIS IN
        # ------------
        batch_size = queries.shape[0]
        q = self.Q(queries.view(batch_size, -1, queries.shape[-1]))
        k = self.K(keys)
        v = self.V(values)
        unnormalized_attention = k@q.transpose(2,1)*self.scaling_factor
        mask = ~torch.triu(unnormalized_attention).bool()
        attention_weights = self.softmax(unnormalized_attention.masked_fill(mask, self.neg_inf))
        context = attention_weights.transpose(2,1)@v
        return context, attention_weights



class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)        
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        self.self_attentions = nn.ModuleList([nn.ModuleList([CausalScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_heads)]) for j in range(self.num_layers)])
        self.encoder_attentions = nn.ModuleList([nn.ModuleList([ScaledDotAttention(
                                    hidden_size=hidden_size, 
                                 ) for i in range(self.num_heads)]) for j in range(self.num_layers)])
        self.attention_mlps = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size),
                                    nn.ReLU(),
                                 ) for i in range(self.num_layers)])

        self.linear_after_causal = nn.Linear(self.num_heads*hidden_size, hidden_size)
        self.linear_after_scaled = nn.Linear(self.num_heads*hidden_size, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)

        self.positional_encodings = self.create_positional_encodings()

        self.dropout = nn.Dropout(p=dropout)

        self.layernorms1 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])
        self.layernorms2 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])
        self.layernorms3 = nn.ModuleList([nn.LayerNorm([self.hidden_size]) for i in range(self.num_layers)])

    def forward(self, inputs, annotations):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: Not used in the transformer decoder
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """
        
        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        # THIS LINE WAS ADDED AS A CORRECTION. 
        embed = embed + self.positional_encodings[:seq_len]

        encoder_attention_weights_list = []
        self_attention_weights_list = []
        contexts = embed

        

        for i in range(self.num_layers):
          # ------------
          # FILL THIS IN - START
          # ------------
            concat_causal = torch.empty((batch_size, seq_len, 0), device='cuda')
            concat_scaled = torch.empty((batch_size, seq_len, 0), device='cuda')
            for j in range(self.num_heads):
                new_contexts, self_attention_weights = self.self_attentions[i][j](contexts, contexts, contexts)  # batch_size x seq_len x hidden_size
                concat_causal = torch.cat((concat_causal, new_contexts), axis=2)

            new_contexts = self.linear_after_causal(concat_causal) #batch_size x seq_len x hidden_size*num_heads -----> batch_size x seq_len x hidden_size
            new_contexts = self.dropout(new_contexts) #dropout
            residual_contexts = self.layernorms1[i](contexts + new_contexts) #add and norm

            for j in range(self.num_heads):
                new_contexts, encoder_attention_weights = self.encoder_attentions[i][j](residual_contexts, annotations, annotations) # batch_size x seq_len x hidden_size
                concat_scaled = torch.cat((concat_scaled, new_contexts), axis=2)
            
            new_contexts = self.linear_after_scaled(concat_scaled) #batch_size x seq_len x hidden_size*num_heads -----> batch_size x seq_len x hidden_size
            new_contexts = self.dropout(new_contexts) #dropout
            residual_contexts = self.layernorms2[i](residual_contexts + new_contexts) #add and norm

            new_contexts = self.attention_mlps[i](residual_contexts)
            new_contexts = self.dropout(new_contexts) #dropout
            contexts = self.layernorms3[i](residual_contexts + new_contexts) #add and norm
          # ------------
          # FILL THIS IN - END
          # ------------
          
            encoder_attention_weights_list.append(encoder_attention_weights)
            self_attention_weights_list.append(self_attention_weights)
          
        output = self.out(contexts)
        encoder_attention_weights = torch.stack(encoder_attention_weights_list)
        self_attention_weights = torch.stack(self_attention_weights_list)
        
        return output, (encoder_attention_weights, self_attention_weights)

    def create_positional_encodings(self, max_seq_len=1000):
      """Creates positional encodings for the inputs.

      Arguments:
          max_seq_len: a number larger than the maximum string length we expect to encounter during training

      Returns:
          pos_encodings: (max_seq_len, hidden_dim) Positional encodings for a sequence with length max_seq_len. 
      """
      pos_indices = torch.arange(max_seq_len)[..., None]
      dim_indices = torch.arange(self.hidden_size//2)[None, ...]
      exponents = (2*dim_indices).float()/(self.hidden_size)
      trig_args = pos_indices / (10000**exponents)
      sin_terms = torch.sin(trig_args)
      cos_terms = torch.cos(trig_args)

      pos_encodings = torch.zeros((max_seq_len, self.hidden_size))
      pos_encodings[:, 0::2] = sin_terms
      pos_encodings[:, 1::2] = cos_terms

      pos_encodings = pos_encodings.cuda()

      return pos_encodings

class EncoderDecoder(nn.Module):
    '''Decode a source transcription into a target transcription
    '''

    def __init__(
            self, encoder_class, decoder_class,
            target_vocab_size, target_sos=-2, target_eos=-1, encoder_hidden_size=512,
            decoder_hidden_size=1024, word_embedding_size=1024, attention_dim=512, cell_type='lstm', decoder_type='rnn', beam_width=4, dropout=0.0,
            transformer_layers=3, num_heads=1):
        '''Initialize the encoder decoder combo
        '''
        super().__init__()
        self.target_vocab_size = target_vocab_size
        self.target_sos = target_sos
        self.target_eos = target_eos
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.word_embedding_size = word_embedding_size
        self.attention_dim = attention_dim
        self.cell_type = cell_type
        self.decoder_type = decoder_type
        self.beam_width = beam_width
        self.dropout = dropout
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.encoder = self.decoder = None
        self.init_submodules(encoder_class, decoder_class)
        
    def init_submodules(self, encoder_class, decoder_class):
        '''Initialize encoder and decoder submodules
        '''
        self.encoder = encoder_class()
        if self.decoder_type == 'rnn':
            self.decoder = decoder_class(self.target_vocab_size, 
                                    self.target_eos, 
                                    self.word_embedding_size, 
                                    self.encoder_hidden_size, 
                                    self.decoder_hidden_size, 
                                    self.attention_dim,
                                    self.cell_type,
                                    self.dropout)
        else:
            self.decoder = decoder_class(self.target_vocab_size, 
                                    self.encoder_hidden_size,
                                    self.transformer_layers,
                                    self.num_heads,
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
        if self.decoder_type == 'rnn':
            for i in range(len(captions)-1):
                cur_ip = captions[i]
                cur_op, h_cur, attention_weights = self.decoder(cur_ip, cur_op, h_cur, h)
                op.append(cur_op)
                total_attention_weights.append(attention_weights)
            return torch.stack(op), torch.stack(total_attention_weights)
        else:
            op, _ = self.decoder(captions[:-1,:].T, h.permute(1,0,2))
            return op
        

    def beam_search(self, h, max_T, on_max):
        '''
        Inputs:
        h: encoder hidden states. #(H*W, batch_size, L) default is (196, batch_size, 2048)
        '''
        # beam search
        assert not self.training
        if self.decoder_type == 'rnn':
            htilde_tm1 = self.decoder.get_first_hidden_state(h) #(batch_size, decoder_hidden_size)
            logpb_tm1 = torch.where(
                torch.arange(self.beam_width, device=h.device) > 0,  # K
                torch.full_like(
                    htilde_tm1[..., 0].unsqueeze(1), -float('inf')),  # k > 0
                torch.zeros_like(
                    htilde_tm1[..., 0].unsqueeze(1)),  # k == 0
            )  # (N, K)
        else:
            random_placeholder = torch.randn(h.shape[1], self.decoder_hidden_size, device=h.device)
            logpb_tm1 = torch.where(
                torch.arange(self.beam_width, device=h.device) > 0,  # K
                torch.full_like(
                    random_placeholder[..., 0].unsqueeze(1), -float('inf')),  # k > 0
                torch.zeros_like(
                    random_placeholder[..., 0].unsqueeze(1)),  # k == 0
            )  # (N, K)
        
        assert torch.all(logpb_tm1[:, 0] == 0.)
        assert torch.all(logpb_tm1[:, 1:] == -float('inf'))
        b_tm1_1 = torch.full_like(  # (t, N, K)
            logpb_tm1, self.target_sos, dtype=torch.long).unsqueeze(0)
        # We treat each beam within the batch as just another batch when
        # computing logits, then recover the original batch dimension by
        # reshaping
        if self.decoder_type == 'rnn':
            htilde_tm1 = htilde_tm1.unsqueeze(1).repeat(1, self.beam_width, 1)
            htilde_tm1 = htilde_tm1.flatten(end_dim=1)  # (N * K, decoder_hidden_size)
            if self.cell_type == 'lstm':
                ctilde_tm1 = self.decoder.ff_init_c(h.mean(axis=0))
                ctilde_tm1 = ctilde_tm1.unsqueeze(1).repeat(1, self.beam_width, 1)
                ctilde_tm1 = ctilde_tm1.flatten(end_dim=1)
                htilde_tm1 = (htilde_tm1, ctilde_tm1)
        h = h.unsqueeze(2).repeat(1, 1, self.beam_width, 1)
        h = h.flatten(1, 2)  # (S, N * K, L)
        v_is_eos = torch.arange(self.target_vocab_size, device=h.device)
        v_is_eos = v_is_eos == self.target_eos  # (V,)
        t = 0
        logits_tm1 = None
        cur_transformer_ip = None
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
            if self.decoder_type == 'rnn':
                E_tm1 = b_tm1_1[-1].flatten()  # (N * K,)
                logits_t, htilde_t, _ = self.decoder(E_tm1, logits_tm1, htilde_tm1, h)#logits: (N * K, V), htilde_t:(N * K, decoder_hid_size)
            else:
                E_tm1 = b_tm1_1[-1].flatten().unsqueeze(1)  # (N * K, 1)
                if cur_transformer_ip == None:
                    cur_transformer_ip = E_tm1
                else:
                    cur_transformer_ip = torch.cat([cur_transformer_ip, E_tm1], axis=1)
                op, _ = self.decoder(cur_transformer_ip, h.permute(1,0,2))
                logits_t = op[:, -1, :]
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
            if self.decoder_type == 'rnn':
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
            else:
                b_t_1, logpb_t = self.update_beam(None, b_tm1_1, logpb_tm1, logpy_t)
                del logits_t, logpy_t, finished
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

        if htilde_t != None:
            if(self.cell_type == 'lstm'):
                temp_ = temp.unsqueeze(-1).expand_as(htilde_t[0])
                b_t_0 = (torch.gather(htilde_t[0], 1, temp_), torch.gather(htilde_t[1], 1, temp_))
            else:
                temp_ = temp.unsqueeze(-1).expand_as(htilde_t)
                b_t_0 = torch.gather(htilde_t, 1, temp_)

            return b_t_0, b_t_1, logpb_t
        else:
            return b_t_1, logpb_t