from __init__ import *
from bert_embedder import BERTEmbedder

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 bert_tokenizer,
                 bert_config,
                 hparams,
                 device):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_config = bert_config
        self.hparams = hparams
        self.device = device
   
        self.dropout = nn.Dropout(hparams.dropout)

        self.lstm = nn.LSTM(self.bert_config.hidden_size,
                            self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
                
        self.lstm_output_dim = self.hparams.hidden_dim * 1 if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2

        self.classifier = nn.Linear(self.bert_config.hidden_size, hparams.num_classes)

        self.CRF = CRF(hparams.num_classes).cuda()
 
    def forward(self, words, labels):
        input_ids, to_merge_wordpieces, attention_mask, token_type_ids = self._prepare_input(words)

        bert_output = self.bert_model.forward(input_ids=input_ids, 
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask)
        
        # we sum the sum of the last four hidden layers (-1 is the hidden states, see point (3) above)
        layers_to_sum = torch.stack([bert_output[-1][x] for x in [-1, -2, -3, -4]], axis=0)
        summed_layers = torch.sum(layers_to_sum, axis=0)
        merged_output = self._merge_embeddings(summed_layers, to_merge_wordpieces)
        
        embedding_bert = pad_sequence(merged_output, batch_first=True, padding_value=0)
        mask = self.padding_mask(labels)
        embedding_bert = self.dropout(embedding_bert)

        X, (h, c) = self.lstm(embedding_bert)
        X = self.dropout(X)
 
        O = self.classifier(embedding_bert)

        if labels==None:
            log_likelihood = -100
        else:
            log_likelihood = self.CRF.forward(O, labels, mask)

        return log_likelihood, O


    def _prepare_input(self, sentences:List[str]):
      input_ids = []
      # we must keep track of which words have been split so we can merge them afterwards
      to_merge_wordpieces = []
      # BERT requires the attention mask in order to know on which tokens it has to attend to
      # padded indices do not have to be attended to so will be 0
      attention_masks = []
      # BERT requires token type ids for doing sequence classification
      # in our case we do not need them so we set them all to 0
      token_type_ids = []
      # we sum 2 cause we have to consider also [CLS] and [SEP] in the sentence length 
      max_len = max([len(self._tokenize_sentence(s)[0]) for s in sentences]) 
      for sentence in sentences:
        encoded_sentence, to_merge_wordpiece = self._tokenize_sentence(sentence)
        att_mask = [1] * len(encoded_sentence)
        att_mask = att_mask + [0] * (max_len - len(encoded_sentence))
        # we pad sentences shorter than the max length of the batch
        encoded_sentence = encoded_sentence + [0] * (max_len - len(encoded_sentence)) 
        input_ids.append(encoded_sentence)
        to_merge_wordpieces.append(to_merge_wordpiece)
        attention_masks.append(att_mask)
        token_type_ids.append([0] * len(encoded_sentence))
      input_ids = torch.LongTensor(input_ids).to(self.device)
      attention_masks = torch.LongTensor(attention_masks).to(self.device)
      token_type_ids = torch.LongTensor(token_type_ids).to(self.device)
      return input_ids, to_merge_wordpieces, attention_masks, token_type_ids


    def _tokenize_sentence(self, sentence:List[str]):
        encoded_sentence = [self.bert_tokenizer.cls_token_id]
        # each sentence must start with the special [CLS] token
        to_merge_wordpiece = []
        # we tokenize a word at the time so we can know which words are split into multiple subtokens
        for word in sentence:
            encoded_word = self.bert_tokenizer.tokenize(word)
            # we take note of the indices associated with the same word
            to_merge_wordpiece.append([i for i in range(len(encoded_sentence)-1, len(encoded_sentence)+len(encoded_word)-1)]) 
            encoded_sentence.extend(self.bert_tokenizer.convert_tokens_to_ids(encoded_word))
        # each sentence must end with the special [SEP] token
        encoded_sentence.append(self.bert_tokenizer.sep_token_id)
        return encoded_sentence, to_merge_wordpiece



    # aggregated_layers has shape: shape batch_size x sequence_length x hidden_size
    def _merge_embeddings(self, aggregated_layers:List[List[float]],
                          to_merge_wordpieces:List[List[int]]):
        merged_output = []
        # first we remove the [CLS] and [SEP] tokens from the output embeddings
        aggregated_layers = aggregated_layers[:, 1:-1 ,:]
        for embeddings, sentence_to_merge_wordpieces in zip(aggregated_layers, to_merge_wordpieces):
            sentence_output = []
            # for each word we retrieve the indices of its subtokens in the tokenized sentence
            for word_to_merge_wordpiece in sentence_to_merge_wordpieces:
                # we average all the embeddings of the subpieces of a word 
                sentence_output.append(torch.mean(embeddings[word_to_merge_wordpiece], axis=0))
            merged_output.append(torch.stack(sentence_output).to(self.device))
        return merged_output

    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding