import ipdb
import json
import sys
import re

pnci_fp = sys.argv[1]
exts_fp = sys.argv[2]
out_fp = sys.argv[3]

pnci_f = open(pnci_fp) # data/openie/pnc_cvd.jsonl
exts_f = open(exts_fp) # data/openie/pnc_cvd.orig_out.pre_score_carb
out_f  = open(out_fp,'w') # 'data/openie/pnc_cvd.orig_exts.integrate_input.jsonl'

ext2sentD, sent2extsD = {}, {}
for line in exts_f:
    sentence, extraction, confidence = line.split('\t')
    sent_key = sentence.replace(' ','')
    ext2sentD[extraction] = sentence
    if sent_key not in sent2extsD:
        sent2extsD[sent_key] = {'sentence': sentence, 'extractions': []}
    sent2extsD[sent_key]['extractions'].append(extraction)

sent2npD = {}
for i, line in enumerate(pnci_f):
    if i==0 or line.startswith('Average:'):
        continue
    fields = line.split('\t')
    inp, pred = fields[0], fields[2]
    sentence, pnc = inp.split('[SEP]')
    sentence, pnc = sentence.strip(), pnc.strip()
    sent_key = sentence.replace(' ','')
    # jline = json.loads(line)
    if sent_key not in sent2npD:
        sent2npD[sent_key] = []
    sent2npD[sent_key].append([pnc, pred, sentence])

extsent2npD = {}
cnt = 0
json_objs = []
no_exts = 0
for sent_key in sent2npD:
    if sent_key not in sent2extsD: # some sentences have zero extractions
        no_exts += 1
        continue
    orig_sentence = sent2extsD[sent_key]['sentence']
    for ext in sent2extsD[sent_key]['extractions']:
        arg2 = ''
        groups = re.findall('<arg1>(.*)</arg1> <rel>(.*)</rel> <arg2>(.*)</arg2>', ext)
        if not len(groups):
            groups = re.findall('<arg1>(.*)</arg1> <rel>(.*)</rel>', ext)
            if not len(groups):
                continue
            arg1, rel = groups[0][0], groups[0][1]
        else:
            arg1, rel, arg2 = groups[0][0], groups[0][1], groups[0][2]
        arg1, rel, arg2 = arg1.strip(), rel.strip(), arg2.strip()
        extsent = f'{arg1} {rel} {arg2}'
        if extsent not in extsent2npD:
            extsent2npD[extsent] = []
        nps = sent2npD[sent_key]
        for elem in nps:
            pnc = elem[0]
            pnci = elem[1]    
            if arg2.startswith(pnc) and 'is None of' not in pnci:
                extsent2npD[extsent].append(pnc)
                json_objs.append({'input': f'{extsent}. [SEP] {pnci}', 'output': f'{ext} [SEP] {orig_sentence} [SEP] {pnc}'}) # full-stop was added for consistency to training data in integrations input
                cnt += 1

for obj in json_objs:
    out_f.write(json.dumps(obj)+'\n')
out_f.close()

print('Cases where extractions have objects starting with noun compounds = ', cnt)
print('Cases where sentence is not present in extractions = ', no_exts)