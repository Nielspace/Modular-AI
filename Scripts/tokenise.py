import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


import tiktoken
import re


class TextDataset(Dataset):
    """
    use pandas dataset
    """
    def __init__(self, articles, model="gpt2", seq_length=512):
        self.tokenizer = tiktoken.get_encoding(model)
        self.vocab_size = self.tokenizer.n_vocab
        self.seq_length = seq_length
        self.articles = articles.apply(self.preprocess_and_tokenize)
        
        self.input_ids, self.attention_masks, self.targets = self.create_sequences()

    def preprocess_and_tokenize(self, text):
        # Preprocess text
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Check for invalid token indices
        assert all(token < self.vocab_size for token in tokens), "Token index out of range"
        
        # Pad and truncate tokens
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        else:
            tokens += [0] * (self.seq_length - len(tokens))
        self.tokens = tokens
            
        return tokens

    def create_sequences(self):
        input_ids = []
        attention_masks = []
        targets = []
        
        for tokens in self.articles:
            input_ids.append(tokens[:-1])  # Exclude the last token for input
            targets.append(tokens[1:])     # Exclude the first token for target
            attention_masks.append([1 if token != 0 else 0 for token in tokens[:-1]])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        
        return input_ids, attention_masks, targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_seq = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        target_seq = self.targets[idx]
        
        sample = {'input_ids': input_seq, 'targets': target_seq, 'attention_mask': attention_mask}
        return sample

def pad_sequences(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    targets = [item['targets'] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)

    return {'input_ids': input_ids_padded, 'targets': targets_padded, 'attention_mask': attention_masks_padded }
