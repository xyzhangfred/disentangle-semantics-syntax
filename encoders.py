import torch
import model_utils
import torch.nn as nn
import re
from pytorch_pretrained_bert.modeling import BertModel,BertConfig, BertEmbeddings, BertEncoder


class encoder_base(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(encoder_base, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))
            log.info(
                "{} initialized with pretrained word embedding".format(
                    type(self)))


class word_avg(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, log,
                 *args, **kwargs):
        super(word_avg, self).__init__(vocab_size, embed_dim, embed_init, log)

    def forward(self, inputs, mask):
        input_vecs = self.embed(inputs.long())
        sum_vecs = (input_vecs * mask.unsqueeze(-1)).sum(1)
        avg_vecs = sum_vecs / mask.sum(1, keepdim=True)
        return input_vecs, avg_vecs


class bilstm(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, hidden_size, log,
                 *args, **kwargs):
        super(bilstm, self).__init__(vocab_size, embed_dim, embed_init, log)
        self.lstm = nn.LSTM(
            embed_dim, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, inputs, mask, temp=None):
        input_vecs = self.embed(inputs.long())
        outputs, _ = model_utils.get_rnn_vecs(
            input_vecs, mask, self.lstm, bidir=True)
        outputs = outputs * mask.unsqueeze(-1)
        sent_vec = outputs.sum(1) / mask.sum(1, keepdim=True)
        #print ('input_vecs.shape', input_vecs.shape)
        #print ('sent_vec.shape', sent_vec.shape)
        return input_vecs, sent_vec


class Bert(encoder_base):
    def __init__(self, vocab_size, embed_dim, embed_init, hidden_size, log,
                 *args, **kwargs):
        super(Bert, self).__init__(vocab_size, embed_dim, embed_init, log)
        #print ('init bert')
        self.config = BertConfig(vocab_size_or_config_json_file=vocab_size,
                 hidden_size=embed_dim,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=4 * hidden_size,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02)
        self.max_length = 128
        self.BertEncoder = BertEncoder(self.config)
        print ('init BertModel')

    def forward(self, inputs, mask, temp=None):
        input_vecs = self.embed(inputs.long())
        
        #print ('hello')

        if input_vecs.shape[1] >= self.max_length:
            input_vecs = input_vecs[:,self.max_length]
            mask = mask[:,self.max_length]
        else:
            input_vecs = torch.cat((input_vecs, \
                                    torch.zeros(input_vecs.shape[0], self.max_length - input_vecs.shape[1],input_vecs.shape[2] ).cuda()
                                    ) , 1)
            mask = torch.cat((mask, \
                                    torch.zeros(mask.shape[0], self.max_length - mask.shape[1]).cuda()
                                    ) , 1)
                
        #breakpoint()
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.BertEncoder(input_vecs,
                                      extended_attention_mask,
                                      output_all_encoded_layers=False)        
        
        
        #breakpoint()
        #print ('bye')
        ###preprocess the input so it fits the Bert format.
        ###Add position embedding and        
        ##zero_padding
        
        encoded_layers = encoded_layers[0] * mask.unsqueeze(-1)
        sent_vec = encoded_layers.sum(1) / mask.sum(1, keepdim=True)
        
        return input_vecs, sent_vec




class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_inputs_to_features(inputs,mask, seq_length):
    """
    convert the original input to bert features, including add positional embedding,
    segment embedding, and zero-padding
    
    
    """

    features = []
    for index,input_ids in enumerate(inputs):
        # Account for [CLS] and [SEP] with "- 2"
        if len(input_ids) > seq_length:
            input_ids = input_ids[0:(seq_length)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        input_type_ids = [0] * len(input_ids)


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

#        if ex_index < 5:
#            logger.info("*** Example ***")
#            logger.info("unique_id: %s" % (example.unique_id))
#            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
#            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#            logger.info(
#                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=index,
                tokens=None,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features
