import torch
import torch.nn as nn
from Semantic_Parser_Executor.Parser.PositionalEncoding import PositionalEncoding
import pickle

# import BERT
from transformers import BertModel

#bert = BertModel.from_pretrained('bert-base-uncased')
bert = pickle.load(open( "model_files/bert.pickle", "rb" ))

class TransformerModel(nn.Module):

    def __init__(self, nreptoken, nhead, nhid, nlayers, dropout=0.5):  # nnertoken
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, \
            Transformer
        self.model_type = 'Transformer'

        embedding_dim = bert.config.to_dict()['hidden_size']
        self.bert = bert
        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        # decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        ##self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(nreptoken, embedding_dim)  # encode to the same dimensionality as bert
        self.bert_encoder = bert
        self.transformer = Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=nlayers,
                                       num_decoder_layers=nlayers, dim_feedforward=nhid, dropout=dropout,
                                       activation='relu', custom_encoder=None, custom_decoder=None)
        self.linear = nn.Linear(embedding_dim, nreptoken)
        self.linear_ner = nn.Linear(embedding_dim, nreptoken)
        # self.decoder = TransformerDecoder(decoder_layers, nlayers)
        # self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #    device = src.device
        #    mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #   self.src_mask = mask
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            device = tgt.device
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            self.tgt_mask = mask

        # get the padding masks
        #src_pad_mask = torch.tensor(src == pad_token_idx)
        #tgt_pad_mask = torch.tensor(tgt == self.pad_id).permute(1, 0)

        # print(src.size())
        # with torch.no_grad(): #try to enable for NER purposes
        src = self.bert_encoder(src)[0]

        src = src.permute(1, 0, 2)  # you need to permute the bert output because it has flipped batch/seq dimensions, otherwise transformer doesn't comsume it correctly
        src = self.pos_encoder(src)
        # print(src.size())

        # tgt = tgt.transpose(0,1)
        # with torch.no_grad():
        tgt = self.encoder(tgt)
        tgt = self.pos_encoder(tgt)

        # output = self.transformer_encoder(src, self.src_mask)
        # print(src.size(),tgt.size())

        # ,src_key_padding_mask = src_pad_mask, tgt_key_padding_mask = tgt_pad_mask
        output = self.transformer(src=src, tgt=tgt, tgt_mask=self.tgt_mask)  #, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask
        output = self.linear(output)
        output_ner = self.linear_ner(src)  # should we add a encoder layer here?, probably no
        output = torch.cat((output, output_ner))
        return output