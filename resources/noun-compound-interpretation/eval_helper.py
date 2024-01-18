from datasets import load_metric

import semeval_scorer

rouge = load_metric('rouge')
meteor = load_metric('meteor')
bleu = load_metric('bleu')
bert = load_metric('bertscore')


def rougeL_f1_scorer(candidate, reference):
    score = rouge.compute(predictions=[candidate], references=[reference], use_aggregator=False)
    return score['rougeL'][0].fmeasure


def meteor_scorer(candidate, reference):
    score = meteor.compute(predictions=[candidate], references=[reference])
    return score['meteor']


def bleu_scorer(candidate, reference):
    score = bleu.compute(predictions=[candidate.split()], references=[[reference.split()]])
    return score  # TODO: return number


def bert_scorer(candidate, reference, model_type='???'):
    score = bert.compute(predictions=[candidate], references=[reference], model_type=model_type)
    return score['f1'][0]


def score_paraphrases_average(paraphrases, ref_paraphrases, scorer):
    total_score = 0

    for para in paraphrases:
        best_match, best_score = score_single_paraphrase(para, ref_paraphrases, scorer)
        total_score += best_score

    average_score = total_score / len(paraphrases)
    return average_score


def score_single_paraphrase(paraphrase, ref_paraphrases, scorer):
    text = paraphrase.lstrip()  # strip leading spaces

    best_score = 0
    best_match = ""
    for ref_paraphrase in ref_paraphrases:
        score = scorer(text, ref_paraphrase)

        if score > best_score:
            best_score = score
            best_match = ref_paraphrase

    return best_match, best_score


def batch_meteor(gen_paraphrases, ref_paraphrases):
    for gen_para in gen_paraphrases:
        for ref_para in ref_paraphrases:
            meteor.add_batch(predictions=[gen_para], references=[ref_para])

    avg_score = meteor.compute()
    return avg_score['meteor']


def batch_meteor_score(gen_paraphrases, ref_paraphrases):
    running_avg = 0
    for gen_para in gen_paraphrases:
        scores = []
        for ref_para in ref_paraphrases:
            score = meteor.compute(predictions=[gen_para], references=[ref_para])
            scores.append(score['meteor'])
        top_score = max(scores)
        running_avg += top_score

    avg_score = running_avg / len(gen_paraphrases)
    return avg_score


def batch_rougel_score(gen_paraphrases, ref_paraphrases):
    for gen_para in gen_paraphrases:
        for ref_para in ref_paraphrases:
            rouge.add_batch(predictions=[gen_para], references=[ref_para])

    rouge_l_scores = rouge.compute(use_aggregator=False)['rougeL']
    rouge_l_f1_scores = [sc.fmeasure for sc in rouge_l_scores]

    running_avg = 0
    for i in range(len(gen_paraphrases)):
        start = i * len(gen_paraphrases)
        end = start + len(gen_paraphrases)
        f1_max = max(rouge_l_f1_scores[start:end])
        running_avg += f1_max

    avg_score = running_avg / len(gen_paraphrases)
    return avg_score


def batch_bertsc_score(gen_paraphrases, ref_paraphrases):
    for gen_para in gen_paraphrases:
        for ref_para in ref_paraphrases:
            bert.add_batch(predictions=[gen_para], references=[ref_para])
    bert_f1_scores = bert.compute(lang='en')['f1']

    running_avg = 0
    for i in range(len(gen_paraphrases)):
        start = i * len(gen_paraphrases)
        end = start + len(gen_paraphrases)
        f1_max = max(bert_f1_scores[start:end])
        running_avg += f1_max

    avg_score = running_avg / len(gen_paraphrases)
    return avg_score


def batch_semval_score(gen_paraphrases, ref_paraphrases):
    avg_score = semeval_scorer.get_paraphrase_score(gen_paraphrases, ref_paraphrases)
    return avg_score


def all_scores(generated_df, reference_df):
    global_total_meteor = 0
    global_total_rougel = 0
    global_total_bertsc = 0
    global_total_semval = 0

    for index, row in generated_df.iterrows():
        print("progress: ", index, "/", len(generated_df))
        target_nc = row['nc']
        gen_paras = row['paras']
        print("scoring: '", target_nc, "'")

        ref_paras = reference_df.loc[reference_df['nc'] == target_nc]['paraphrases'].iloc[0]

        meteor_score = batch_meteor_score(gen_paras, ref_paras)
        rougel_score = batch_rougel_score(gen_paras, ref_paras)
        bertsc_score = batch_bertsc_score(gen_paras, ref_paras)
        semval_score = batch_semval_score(gen_paras, ref_paras)

        global_total_meteor += meteor_score
        global_total_rougel += rougel_score
        global_total_bertsc += bertsc_score
        global_total_semval += semval_score

    num_paras = len(generated_df)

    global_average_meteor = global_total_meteor / num_paras
    global_average_rougel = global_total_rougel / num_paras
    global_average_bertsc = global_total_bertsc / num_paras
    global_average_semval = global_total_semval / num_paras

    print("global meteor, ", global_average_meteor)
    print("global rougel, ", global_average_rougel)
    print("global bertsc, ", global_average_bertsc)
    print("global semval, ", global_average_semval)

    return global_average_meteor, global_average_rougel, global_average_bertsc, global_average_semval
