from __init__ import *
from dataset import IdiomDataset
from collate import collate
from model import IdiomExtractor
from hparams import HParams
from trainer import Trainer
from utils import *


if __name__=="__main__":
    #we set a seed for having replicability of results
    SEED = 2
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    #instantiate bert
    model_name = 'bert-base-multilingual-cased'
    bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name, config=bert_config)

    modality = input("Do you want to train or test your system? ")

    language = input("Select a language (e.g., English): ")
    admitted_languages = ["English", "Italian", "French", "Spanish", "Dutch", "German", "Polish", "Portuguese", "Chinese", "Japanese"]
    assert language in admitted_languages, f"Not a valid language.\nPlease choose one language from the following list: {(', ').join(admitted_languages)}.\n\n"

    spacy_tagger = initialize(language)

    if modality == "test":
        checkpoint = input("Insert the name of your model checkpoint (it should be placed into ./src/checkpoints/): ")

    #resources
    #train_file = "./resources/bio_format/English_idiom_expr_identification.tsv"
    train_file = f"./resources/bio_format/{language.lower()}/train_{language.lower()}.tsv"
    dev_file = f"./resources/bio_format/{language.lower()}/dev_{language.lower()}.tsv"
    test_file = f"./resources/bio_format/{language.lower()}/test_{language.lower()}.tsv"

    labels_vocab = {"<pad>":0, "B-IDIOM":1, "I-IDIOM":2, "O":3}


    #index dataset
    train_dataset = IdiomDataset(train_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")
    dev_dataset = IdiomDataset(dev_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")
    test_dataset = IdiomDataset(test_file, bert_tokenizer, labels_vocab, spacy_tagger, "all")

    '''train_dataset = []
    for entry in train_dataset_tmp:
        present = False
        for entry2 in test_dataset:
            if entry[0] == entry2[0]:
                present = True
        
        if present==False:
            train_dataset.append(entry)'''

    #test_dataset = IdiomDataset("./resources/bio_format/test.tsv", bert_tokenizer, labels_vocab) #individual samples test
    print(f"train sentences: {len(train_dataset)}")
    print(f"dev sentences: {len(dev_dataset)}")
    print(f"test sentences: {len(test_dataset)}")

    idioms_train = get_idioms(train_dataset, spacy_tagger)
    idioms_dev = get_idioms(dev_dataset, spacy_tagger)
    idioms_test = get_idioms(test_dataset, spacy_tagger)

    percentage_elements_in_train_also_in_dev = overlap_percentage_l1_in_l2(idioms_dev, idioms_train) #1/2512
    percentage_elements_in_train_also_in_test = overlap_percentage_l1_in_l2(idioms_test, idioms_train) #1/2512
    print(percentage_elements_in_train_also_in_dev, percentage_elements_in_train_also_in_test)

    if modality == "test":
        dataset_type = input("Do you want to test your system on all sentences, the seen ones or the unseen ones? ")
        test_dataset = IdiomDataset(test_file, bert_tokenizer, labels_vocab, spacy_tagger, dataset_type, idioms_train, idioms_test)
        print(f"test sentences {dataset_type}: {len(test_dataset)}")

    print(len(test_dataset))

    #dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate)
    print(len(train_dataloader))
    print(len(dev_dataloader))
    print(len(test_dataloader))


    #instantiate the hyperparameters
    params = HParams()


    #instantiate the model
    my_model = IdiomExtractor(bert_model, 
                        bert_tokenizer, 
                        bert_config, 
                        params,
                        "cuda").cuda()

    if modality=="test": 
        my_model.load_state_dict(torch.load(f"./src/checkpoints/{checkpoint}"))
    
    print(my_model)


    #trainer
    trainer = Trainer(model = my_model,
                    loss_function = nn.CrossEntropyLoss(ignore_index=0),
                    optimizer = optim.Adam(bert_model.parameters(), lr=0.00001),
                    labels_vocab=labels_vocab,
                    gradient_accumulation_steps=4)

    if modality == "train":
        trainer.train(train_dataloader, dev_dataloader, 100, patience=5, modelname = f"{language.lower()}")

    else:
        trainer.evaluate(test_dataloader, "test")

    


