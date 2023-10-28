from seqeval.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score
import sys
import os
import json

if __name__ == "__main__":
    results_path = sys.argv[1]  # model path with predictions
    results_mode = sys.argv[2]  # test or dev

    out_slot_label_list = json.load(
        open(os.path.join(results_path, "out_slot_label_list_%s_0.json" % (results_mode)), "r"))
    slot_preds_list = json.load(open(os.path.join(results_path, "slot_preds_list_%s_0.json" % (results_mode)), "r"))
    with open(os.path.join(results_path, "classification_report_%s_0.txt" % (results_mode)), "w") as fout:
        fout.write(classification_report(out_slot_label_list, slot_preds_list, digits=4))
    print(precision_score(out_slot_label_list, slot_preds_list))
    print(recall_score(out_slot_label_list, slot_preds_list))
    print(f1_score(out_slot_label_list, slot_preds_list))
