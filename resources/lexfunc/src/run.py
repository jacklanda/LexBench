from collections import defaultdict,Counter
import numpy as np
import sys
import os
import gensim
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as report
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from config import params

def load_embeddings(embeddings_path):
    print('Loading embeddings:',embeddings_path)
    try:
        model=gensim.models.Word2Vec.load(embeddings_path)
    except:
        try:
            model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path)
        except:
            try:
                model=gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,binary=True)
            except:
                sys.exit('Couldnt load embeddings')
    vocab=model.vocab
    dims=model.vector_size
    vocab=set(vocab)
    return model,vocab,dims

def print_summary(pairs):
    print('== LABEL SUMMARY ==')
    for l in pairs:
        count=len(pairs[l])
        print('Using label: ',l,' | With freq: ',count)
    print('== LABEL SUMMARY ==\n')

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

# data paths
diffvec_file = params['diffvec_file']
collocations_folder = params['collocations_folder']
word_vectors = params['word_vectors']
relation_vectors = params['relation_vectors']

# minimum label frequency - set to 0 to use full dataset unrestricted
min_label_freq = params['min_label_freq']

# vector composition operation - Choose from ['concat','diff','sum','mult','firstw']
composition = params['composition']

# which data to use - Choose list from ['diffvec', 'collocations']
data_choice = params['data_choice']

# concatenate relation vector to the word embedding composition?
USE_SEVEN=params['USE_SEVEN']

# black list of morphosyntactic labels from diffvec
#MORPHOSYNTACTIC_LABELS=params['morphosyntactic_labels']

# load embeddings
modelwords,vocabwords,dimwords=load_embeddings(word_vectors)
if USE_SEVEN:
    modelrels,vocabrels,dimrels=load_embeddings(relation_vectors)

all_pairs = defaultdict(list)
all_labels=set()
lcount=defaultdict(int)

for d in data_choice:
    if d=='diffvec':
        ### load diffvec ###
        for line in open(diffvec_file,'r'):
            line=line.strip().lower()
            elms=line.split(',')
            label=elms[0]
            #if not label in MORPHOSYNTACTIC_LABELS:
            w1,w2=elms[1],elms[2]
            if w1 in vocabwords and w2 in vocabwords:
                all_pairs[label].append((w1,w2))
                lcount[label]+=1
                all_labels.add(label)
    else:
        ### LOAD COLLOCATIONS ###
        for infile in os.listdir(collocations_folder):
            label=infile[:infile.index('.txt_')]
            with open(os.path.join(collocations_folder,infile),'r') as f:
                for line in f:
                    cols=line.lower().strip().split('\t')
                    base,col=cols[0],cols[1]
                    all_pairs[label].append((base,col))
                    all_labels.add(label)
                    lcount[label]+=1

# filter by min frequency
filtered_pairs = defaultdict(list)
for label in all_labels:
    if lcount[label] >= int(min_label_freq):
        for pair in all_pairs[label]:
            filtered_pairs[label].append(pair)

print('Originally there were ',len(all_pairs),' labels')
print('Now there were ',len(filtered_pairs),' labels')
print()

print_summary(filtered_pairs)

# vectorize and split
X=[]
y=[]
label2id={}

for label in filtered_pairs:
    for w1,w2 in filtered_pairs[label]:
        if not label in label2id:
            label_id=len(label2id)
            label2id[label]=label_id
        rel=w1+'__'+w2
        found=False
        if w1 in modelwords and w2 in modelwords:
            if composition == 'concat':
                xi=np.concatenate([modelwords[w1],modelwords[w2]])
            elif composition == 'diff':
                xi=modelwords[w1]-modelwords[w2]
            elif composition == 'sum':
                xi=modelwords[w1]+modelwords[w2]
            elif composition == 'mult':
                xi=modelwords[w1]*modelwords[w2]
            elif composition == 'leftw':
                xi=modelwords[w2]
            found=True
        if found and USE_SEVEN:
            if rel in vocabrels:
                xi=np.concatenate([xi,modelrels[rel]])
            else:
                xi=np.concatenate([xi,np.zeros(dimrels)])
        if found:
            X.append(xi)
            y.append(label_id)

X=np.array(X)
y=np.array(y)

print('Data tensor shape: ',X.shape)
print('Label tensor shape: ',y.shape)
print('Params:')
for a,b in params.items():
    print(a,'--> ',b)

if params['experiment'] == 'cv':

    # K-FOLD CV
    numb_splits = 10
    kf = KFold(n_splits=numb_splits,random_state=10, shuffle=True)

    fold_count=1
    all_scores=defaultdict(list)
    for train_index, test_index in kf.split(X):
        X_train=X[train_index]
        y_train=y[train_index]
        X_test=X[test_index]
        y_test=y[test_index]
        clf=LinearSVC()
        print('For fold number ',fold_count,' - Training SVM...')
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        prec_results=precision(y_test,pred, average='macro')
        rec_results=recall(y_test,pred, average='macro')
        f1_results=f1(y_test,pred, average='macro')
        accres=accuracy_score(y_test,pred)
        print('For fold ',fold_count)
        print('precision: ',prec_results)
        print('recall:',rec_results)
        print('f1:',f1_results)
        print('acc:',accres)
        print('---')
        all_scores['precision'].append(prec_results)
        all_scores['recall'].append(rec_results)
        all_scores['f1'].append(f1_results)
        all_scores['accuracy'].append(accres)
        fold_count+=1

    report=[]
    print('=== FINAL SCORES ===')
    for i in all_scores:
        avg_res=sum(all_scores[i])/len(all_scores[i])
        print(i,'->',avg_res)
        report.append(i+'_'+str(avg_res))

else:

    X,y=shuffle(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=4, stratify=y)

    final_label_ids=[]
    final_label_names=[]
    for label,idx in label2id.items():
        final_label_ids.append(idx)
        final_label_names.append(label)

    clf=LinearSVC()
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)    

    print(classification_report(y_test,pred,labels=final_label_ids,target_names=final_label_names))
    print('Precision: ',np.average(report(y_test,pred)[0]))
    print('Recall: ',np.average(report(y_test,pred)[1]))
    print('F1: ',np.average(report(y_test,pred)[2]))
    print('Accuracy: ',accuracy_score(y_test,pred))

"""
# generate confusion matrix from the last fold (cv) or the test set (split)
cm=confusion_matrix(y_test,pred)
normalized=cm / cm.astype(np.float).sum(axis=1, keepdims=True)
df_cm = pd.DataFrame(normalized, index = label2id, columns = label2id)
fig, ax = plt.subplots(figsize=(10,7))         # Sample figsize in inches
sns.heatmap(df_cm, annot=False, cmap="YlGnBu")
plt.show()
"""