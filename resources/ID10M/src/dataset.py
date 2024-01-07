from __init__ import *

class IdiomDataset(Dataset):
    def __init__(self, dataset, tokenizer, labels_vocab, spacy_tagger, type = "all", idioms_train = None, idioms_test = None): # contexts_vocab = contexts_vocab, jobs_vocab = jobs_vocab, parties_vocab = parties_vocab, subjects_vocab = subjects_vocab, states_vocab = states_vocab
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.labels_vocab = labels_vocab
        self.spacy_tagger = spacy_tagger
        self.type = type
        self.idioms_train = idioms_train
        self.idioms_test = idioms_test

        self.sentences = self.get_sentences()
        self.encoded_data = []
        self.encode_data()

    def encode_data(self):
        
        for sentence in tqdm(self.sentences):
            words = []
            labels = []
            idiom = ""
            all_O = True
            for elem in sentence:
                if re.search("\w", elem["token"])!=None or re.search("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~£−€\¿]+", elem["token"])!=None:
                    words.append(elem["token"])
                else:
                    words.append("UNK")
                labels.append(elem["tag"])

                if elem["tag"] == "B-IDIOM":
                    idiom += self.spacy_tagger(elem["token"])[0].lemma_
                    all_O = False
                elif elem["tag"] == "I-IDIOM":
                    idiom += self.spacy_tagger(elem["token"])[0].lemma_

            
            if self.type == "all":
                vectorized_labels = [self.labels_vocab[label] for label in labels]
                encoded_labels = torch.tensor(vectorized_labels)
                self.encoded_data.append((words, encoded_labels))

            elif self.type == "seen":
                if idiom in self.idioms_train:
                    vectorized_labels = [self.labels_vocab[label] for label in labels]
                    encoded_labels = torch.tensor(vectorized_labels)
                    self.encoded_data.append((words, encoded_labels))
            
            elif self.type == "unseen":
                if idiom not in self.idioms_train:
                    vectorized_labels = [self.labels_vocab[label] for label in labels]
                    encoded_labels = torch.tensor(vectorized_labels)
                    self.encoded_data.append((words, encoded_labels))



    def vectorize_words(self, input_vector, special_tokens=True) -> list:
        encoded_words = self.tokenizer.encode(input_vector, add_special_tokens = special_tokens)

        return encoded_words


    def get_sentences(self):
        sentences = []
        sentence = []

        with open(self.dataset, "r") as f:
            for line in f:
                if line!="\n": 
                    line = line.strip().split("\t")
                    token = line[0]
                    tag = line[1]
                    elem = {"token": token, "tag":tag}
                    sentence.append(elem)
                else:
                    sentences.append(sentence)
                    sentence = []
        
        return sentences


    def __len__(self):
        return len(self.encoded_data)
 
    def __getitem__(self, idx:int):
        return self.encoded_data[idx]
    