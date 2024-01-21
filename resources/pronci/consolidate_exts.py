import os
import sys
import ipdb

new_exts_fp = sys.argv[1]
orig_exts_fp = sys.argv[2]
final_exts_fp = sys.argv[3]

new_extsD = {}
for line in open(new_exts_fp):
    line = line.strip()
    sentence = line.split('\t')[0]
    if sentence not in new_extsD:
        new_extsD[sentence] = []
    new_extsD[sentence].append(line)

orig_extsD = {}
for line in open(orig_exts_fp):
    line = line.strip()
    sentence = line.split('\t')[0]
    if sentence not in orig_extsD:
        orig_extsD[sentence] = []
    orig_extsD[sentence].append(line)

final_extsD = {}
for sentence in orig_extsD:
    final_extsD[sentence] = orig_extsD[sentence]
    if sentence in new_extsD:
        final_extsD[sentence].extend(new_extsD[sentence])

with open(final_exts_fp, 'w') as writer:
    for sentence in final_extsD:
        writer.write('\n'.join(final_extsD[sentence])+'\n')