import model
import math
from __init__ import *

class Trainer():
    def __init__(self,
                model:nn.Module, 
                loss_function,
                optimizer,
                labels_vocab,
                gradient_accumulation_steps):
        
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.labels_vocab = labels_vocab
        self.gradient_accumulation_steps = gradient_accumulation_steps
 
    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding
 
    def train(self,
            train_dataset:Dataset, 
            valid_dataset:Dataset,
            epochs:int=1,
            patience:int=10,
            modelname="idiom_expr_detector"):
        
        print("\nTraining...")
 
        train_loss = 0.0
        total_loss_train = []
        total_loss_dev = []
        record_dev = 0.0
        
        full_patience = patience
        
        modelname = modelname
 
        first_epoch = True

        for epoch in range(epochs):
             if patience>0:
                print(" Epoch {:03d}".format(epoch + 1))

                epoch_loss = 0.0
                self.model.train()
                
                count_batches = 0
                self.optimizer.zero_grad()
                
                for words, labels in tqdm(train_dataset):
                    count_batches+=1
                    batch_loss = 0.0

                    batch_LL, _ = self.model(words, labels)
                    batch_NLL = - torch.sum(batch_LL)/8

                    if not math.isnan(batch_NLL.tolist()):
                        batch_NLL.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        epoch_loss += batch_NLL.tolist()

                    if count_batches % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()


                    '''predictions = self.model(words)                     
                    predictions = predictions.view(-1, predictions.shape[-1])
                    labels = labels.view(-1)

                    batch_loss = self.loss_function(predictions, labels)
                    if not math.isnan(batch_loss):
                        batch_loss.backward()
                        epoch_loss += batch_loss.tolist()

                    if count_batches % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()'''


                avg_epoch_loss = epoch_loss / len(train_dataset)
                print('[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, avg_epoch_loss))

                valid_loss, f1 = self.evaluate(valid_dataset)

                if f1>record_dev:
                    record_dev = f1
                    torch.save(self.model.state_dict(), "./src/checkpoints/"+modelname+".pt")
                    patience = full_patience
                else:
                    patience -= 1
                   
                print('\t[E: {:2d}] valid loss = {:0.4f}, f1-score = {:0.4f}, patience: {:2d}'.format(epoch+1, valid_loss, f1, patience))


        print("...Done!")
        return avg_epoch_loss
 

    def evaluate(self, valid_dataset, split="dev"):

        valid_loss = 0.0
        all_predictions = list()
        all_labels = list()
        labels_vocab_reverse = {v:k for (k,v) in self.labels_vocab.items()}
         
        self.model.eval()
    
        for words, labels, in tqdm(valid_dataset):
            batch_loss = 0.0
            self.optimizer.zero_grad()
            
            '''with torch.no_grad():
                predictions = self.model(words)                    

            predictions = predictions.view(-1, predictions.shape[-1])
            
            labels = labels.view(-1)       
            batch_loss = self.loss_function(predictions, labels)'''

            with torch.no_grad():
                batch_LL, predictions = self.model(words, labels)

            batch_NLL = - torch.sum(batch_LL)/8

            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1) 

 
            for i in range(len(predictions)):
                if labels[i]!=0:
                    all_predictions.append(labels_vocab_reverse[int(torch.argmax(predictions[i]))])
                    all_labels.append(labels_vocab_reverse[int(labels[i])])
            
            if not math.isnan(batch_NLL.tolist()):
                valid_loss += batch_NLL.tolist()

        f1 = f1_score(all_labels, all_predictions, average= 'macro')
        print(classification_report(all_labels, all_predictions, digits=3))
        #print(f1)
        
        return valid_loss / len(valid_dataset), f1