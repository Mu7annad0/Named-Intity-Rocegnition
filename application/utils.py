from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def transform_ner_output(ner_output):
    transformed_output = []
    current_entity = None
    current_words = []
    current_start = None

    for i in range(len(ner_output)):
        item = ner_output[i]
        entity = item['entity'][2:]
        word = item['word']
        
        if item['entity'].startswith('B-'):
            if current_entity:
                transformed_output.append({
                    'entity': current_entity,
                    'word': ''.join(current_words).replace('▁', ' ').strip(),
                    'start': current_start,
                    'end': ner_output[i - 1]['end']
                })
            current_entity = entity
            current_words = [word]
            current_start = item['start']+1
        elif item['entity'].startswith('I-') and entity == current_entity:
            current_words.append(word)
    
    if current_entity:
        transformed_output.append({
            'entity': current_entity,
            'word': ''.join(current_words).replace('▁', ' ').strip(),
            'start': current_start,
            'end': ner_output[-1]['end']
        })

    return transformed_output


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device