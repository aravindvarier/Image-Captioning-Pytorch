import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


def compute_average_bleu_over_dataset(model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences
    '''
    with torch.no_grad():
        total_score = 0
        total_num = 0
        for data in tqdm(dataloader):
            torch.cuda.empty_cache()
            images, captions_ref, cap_lens = data
            captions_ref = pad_sequence(captions_ref, padding_value=target_eos)
            images = images.to(device)
            total_num += len(cap_lens)
            b_1 = model(images, on_max='halt')
            captions_cand = b_1[..., 0]
            batch_score = compute_batch_total_bleu(captions_ref, captions_cand, target_sos, target_eos)
            total_score += batch_score

        return total_score/total_num
    
def compute_batch_total_bleu(captions_ref, captions_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch
    '''
    with torch.no_grad():
        refs = captions_ref.T
        cands = captions_cand.T
        refs_list = refs.tolist()
        cands_list = cands.tolist()
        for i in range(len(refs_list)): #Removes sos tags
            refs_list[i] = list(filter((target_sos).__ne__, refs_list[i]))
            cands_list[i] = list(filter((target_sos).__ne__, cands_list[i]))
            
        for i in range(len(refs_list)): #Removes eos tags
            refs_list[i] = list(filter((target_eos).__ne__, refs_list[i]))
            cands_list[i] = list(filter((target_eos).__ne__, cands_list[i]))

        total_score = 0
        for i in range(refs.shape[0]):
            ref = refs_list[i]
            cand = cands_list[i]
            score = BLEU_score(ref, cand, 4)
            total_score += score
        return total_score


def grouper(seq, n):
    '''Extract all n-grams from a sequence
    '''
    ngrams = []
    for i in range(len(seq) - n + 1):
        ngrams.append(seq[i:i+n])
    
    return ngrams


def n_gram_precision(reference, candidate, n):
    '''Calculate the precision for a given order of n-gram
    '''
    total_matches = 0
    ngrams_r = grouper(reference, n)
    ngrams_c = grouper(candidate, n)
    total_num = len(ngrams_c)
    assert total_num > 0
    for ngram_c in ngrams_c:
        if ngram_c in ngrams_r:
            total_matches += 1
    return total_matches/total_num
    


def brevity_penalty(reference, candidate):
    '''Calculate the brevity penalty between a reference and candidate
    '''
    if len(candidate) == 0:
        return 0
    if len(reference) <= len(candidate):
        return 1
    return np.exp(1 - (len(reference)/len(candidate)))



def BLEU_score(reference, hypothesis, n):
    '''Calculate the BLEU score
    '''
    bp = brevity_penalty(reference, hypothesis)
    prec = 1
    cand_len = min(n, len(hypothesis))
    if(cand_len == 0):
    	  return 0
    for i in range(1, cand_len + 1):
        prec = prec * n_gram_precision(reference, hypothesis, i)
    prec = prec ** (1/n)
    return bp * prec