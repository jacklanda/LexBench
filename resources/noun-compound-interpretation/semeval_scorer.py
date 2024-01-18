# SemEval 2013 Task 4 Non-Isomorphic Scoring
min_match_score = 0.1  # minimum match value for two different ngrams
min_stem_len = 3  # minimum shared prefix length for two Strings to match


def get_paraphrase_score(test_paras, ref_paras):
    total_score = 0
    for test_para in test_paras:
        best_match, score = get_best_match(test_para, ref_paras)

        if best_match is None:
            print("no match for: ", test_para)
            continue

        total_score += score

    average_score = total_score / len(test_paras)
    return average_score


def get_best_match(test_p, ref_paras):
    best_match = None
    best_score = 0

    for ref_p in ref_paras:
        score = get_match(test_p, ref_p)

        if score > best_score:
            best_score = score
            best_match = ref_p

    return best_match, best_score


def get_match(test_p, ref_p):
    match = sum_shared_1_to_ngrams(test_p, ref_p)

    if match < 0.01:
        return match

    maximum = max(sum_shared_1_to_ngrams(ref_p, ref_p), sum_shared_1_to_ngrams(test_p, test_p))

    if maximum < 0.01:
        return 0.0

    return match / maximum


def sum_shared_1_to_ngrams(p1, p2):
    l1 = p1.split()
    l2 = p2.split()

    max_n = min(len(l1), len(l2))

    total_score = 0
    for n in range(max_n):
        score = sum_shared_ngrams(l1, l2, n)

        if n == 1 and score == 0:
            return 0  # paraphrases don't share any terms

        total_score += score

    return total_score


def sum_shared_ngrams(l1, l2, n):
    max_pos_1 = len(l1) - n
    max_pos_2 = len(l2) - n

    count_matches = 0
    total_score = 0.0
    for i in range(max_pos_1):
        best_score = 0
        best_pos_2 = -1

        for j in range(max_pos_2):
            score = match_ngram(l1, l2, i, j, n)

            if score > best_score:
                best_score = score
                best_pos_2 = j

        if best_pos_2 >= 0:
            total_score += best_score

            if best_score > min_match_score:
                count_matches += 1

    if n == 1 and count_matches < 3:
        return 0.0
    else:
        return total_score


def match_ngram(l1, l2, start1, start2, n):
    total_score = 0.0
    for i in range(n):
        word1 = l1[start1 + i]
        word2 = l2[start2 + i]

        word_match_score = match_words(word1, word2)

        if word_match_score > min_match_score:
            total_score += word_match_score

    return total_score


def match_words(w1, w2):
    if w1 is None or w2 is None:
        return 0.0

    if w1 == w2:
        return 1.0  # perfect match

    if len(w1) < min_stem_len or len(w2) < min_stem_len:
        return 0.0  # not long enough to match reliably

    if w1[0:min_stem_len] != w2[0:min_stem_len]:
        return 0.0  # no shard stem, no match

    overlap = 0

    for i in range(len(w1)):
        if i >= len(w2):
            break

        if w1[i] == w2[i]:
            overlap += 1
        else:
            break

    overlap_score = (2 * overlap) / (len(w1) + len(w2))

    return overlap_score * overlap_score  # punish imperfect matches
