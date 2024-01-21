import spacy
import json
import ipdb
import sys

from tqdm import tqdm

inp_fp = sys.argv[1] # 'data/openie/cvd_news.txt'
out_fp = sys.argv[2]

nlp = spacy.load("en_core_web_sm")
noun_compounds = {'mwe': [], 'pnc': [], 'cnc': [], 'oth': []}
for line in tqdm(open(inp_fp).readlines()):
    line = line.strip()
    doc = nlp(line)
    for tok in [tok for tok in doc if tok.dep_ == 'compound']:
        head = tok.head

        mwe = False
        if head.i - tok.i > 1 and doc[tok.i+1].dep_ != 'punct':
            mwe = True
        else:
            while head.dep_ == 'compound':
                head = head.head

        noun = doc[tok.i: head.i + 1]
        noun_json = {'nnp': noun[:-1].text, 'nn': noun[-1].text, 
        'explicit_relation': noun.text, 'sentence': line,
        'nnp_index': 0, 'nn_index': 1, 'is_compositional': True, 'comment': 'None', 
        'relation': [""], 'is_context_needed': True, 'worker_id': ""}

        if mwe:
            noun_compounds['mwe'].append(json.dumps(noun_json))
        elif noun[-1].pos_ == 'NOUN':
            if sum([n.pos_ != 'PROPN' for n in noun[:-1]]):
                noun_compounds['cnc'].append(json.dumps(noun_json))
            else:
                noun_compounds['pnc'].append(json.dumps(noun_json))
        else:
            noun_json['comment'] = ' '.join([str(t.pos_) for t in noun])
            noun_compounds['oth'].append(json.dumps(noun_json))

with open(out_fp,'w') as writer:
    writer.write('\n'.join(noun_compounds['pnc']))
# open('data/openie/mwe_cvd.jsonl','w').write('\n'.join(noun_compounds['mwe']))
# open('data/openie/pnc_cvd.jsonl','w').write('\n'.join(noun_compounds['pnc']))
# open('data/openie/cnc_cvd.jsonl','w').write('\n'.join(noun_compounds['cnc']))
# open('data/openie/oth_cvd.jsonl','w').write('\n'.join(noun_compounds['oth']))