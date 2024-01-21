import sys
import ipdb
import json

import regex as re

inp_fp = sys.argv[1]
out_fp = sys.argv[2]

lines = []
for line in open(inp_fp):
    jline = json.loads(line)
    int_ext, initial_input = jline['output'], jline['initial_input']
    int_ext = int_ext.strip('.') # full-stop was added for consistency to training data in integrations input
    orig_ext, sentence, pnc = initial_input.split('[SEP]')
    orig_ext, sentence, pnc = orig_ext.strip(), sentence.strip(), pnc.strip()
    pnc_words = pnc.split()
    pn, cn = ' '.join(pnc_words[:-1]), pnc_words[-1]
    arg2 = ''
    groups = re.findall('<arg1>(.*)</arg1> <rel>(.*)</rel> <arg2>(.*)</arg2>', orig_ext)
    if not len(groups):
        groups = re.findall('<arg1>(.*)</arg1> <rel>(.*)</rel>', orig_ext)
        if not len(groups):
            continue
        arg1, rel = groups[0][0], groups[0][1]
    else:
        arg1, rel, arg2 = groups[0][0], groups[0][1], groups[0][2]
    arg1, rel, arg2 = arg1.strip(), rel.strip(), arg2.strip()
    try:
        start_index = int_ext.index(arg1+' '+rel)+len(arg1+' '+rel)
        end_index = int_ext.index(pn)
    except:
        continue
    new_rel = rel + ' ' + int_ext[start_index:end_index].strip()
    new_rel = new_rel.strip()
    new_arg2 = int_ext[end_index:]
    new_arg2 = new_arg2.strip()

    new_ext = f'<arg1> {arg1} </arg1> <rel> {new_rel} </rel> <arg2> {new_arg2} </arg2>'
    lines.append(f'{sentence}\t{new_ext}\t1')

with open(out_fp,'w') as writer:
    writer.write('\n'.join(lines))