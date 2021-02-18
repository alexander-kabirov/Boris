import torch
from Semantic_Parser_Executor.Parser.TransformerModule import TransformerModel
import pickle
import numpy as np
from transformers import BertTokenizer


class Model:
  def __init__(self,model_name='model_files/model_14062020.state',nhid=300,nlayers = 2,nhead = 4,dropout = 0.1):
    #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.tokenizer = pickle.load(open("model_files/tokenizer.pickle", "rb"))
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #self.tokenize = lambda x: self.tokenizer.tokenize(x)
    self.vocab = pickle.loads(open('model_files/vocab.pickle', 'rb').read())
    nreptokens = len(self.vocab.stoi)
    self.model = TransformerModel(nreptokens, nhead, nhid, nlayers, dropout).to(self.device)
    self.model.load_state_dict(torch.load(model_name))
    print('Loaded model, # of loaded parameters is',sum([p.numel() for p in self.model.parameters()]))

  def generate(self,src,max_length=100):  # src is has the [seq_l] dimensionality
    # get <pad> id
    pad_id = self.vocab.stoi['<pad>']  # REP
    eos_id = self.vocab.stoi['<eos>']  # REP
    sos_id = self.vocab.stoi['<sos>']  # REP
    init_tgt = [sos_id]
    eos_counter = 0
    output_ner = []
    output_rep = []
    for i in range(max_length):
      tgt_input = init_tgt + [pad_id] * (max_length - len(src))
      tgt_input = torch.tensor(tgt_input).to(self.device)
      # tgt_input = torch.tensor(list(REP.numericalize([tgt_input]))).to(device)
      output = self.model(src.view(1, len(src)), tgt_input.view(len(tgt_input), 1))
      output_ner = output[-len(src):, 0, :]
      output_rep = output[:-len(src), 0, :]
      _, max_ind = torch.max(output_rep, 1)
      rep = [self.vocab.itos[i] for i in max_ind]
      init_tgt = init_tgt + [max_ind[i]]
      if max_ind[i] == eos_id:
        break
    _, max_ind = torch.max(output_ner, 1)
    output_ner_text = [self.vocab.itos[i] for i in max_ind]
    return ([self.vocab.itos[i] for i in init_tgt], output_ner_text)

  def parse(self,text):
    src_input = self.tokenizer.tokenize(text) #self.tokenize
    src_input = ['[CLS]'] + src_input + ['[SEP]']
    src_input = self.tokenizer.convert_tokens_to_ids(src_input)
    src_input = torch.tensor(src_input).to(self.device)
    output = self.generate(src_input)
    ner_output = np.array(output[1])
    entity_values = list(ner_output[ner_output != 'o'][1:-1])  # cut sos and eos
    entity_indexes = list(np.where(ner_output != 'o')[0][1:-1])
    key_tokens = {}
    for i, index in enumerate(list(entity_indexes)):
      token_list = key_tokens.get(entity_values[i], [])
      token_list.append(index)
      key_tokens[entity_values[i]] = token_list
    for key in key_tokens:
      tokens = [src_input.cpu().numpy()[i] for i in key_tokens[key]]
      token_str = self.tokenizer.decode(tokens).replace(' _ ', '_')  # join words if underscore is between them
      key_tokens[key] = token_str
    return output[0],key_tokens