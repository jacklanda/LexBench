from __init__ import * 

def collate(elems: tuple) -> tuple:
    words, labels = list(zip(*elems))
    pad_labels = pad_sequence(labels, batch_first=True, padding_value=0)
 
    return list(words), pad_labels.cuda()
