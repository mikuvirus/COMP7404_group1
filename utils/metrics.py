from jiwer import wer, cer

def calculate_wer(reference_texts, hypothesis_texts):

    return wer(reference_texts, hypothesis_texts)

def calculate_cer(reference_texts, hypothesis_texts):

    return cer(reference_texts, hypothesis_texts)
