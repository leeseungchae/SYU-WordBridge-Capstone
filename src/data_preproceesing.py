import json
from torch.utils.data import Dataset
from src.utils import json_loads


class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, src_lang='en', tgt_lang='ko', max_length=128):
        
        self.data = json_loads(file_path)['data']
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while idx < len(self.data):
            item = self.data[idx]
            source_text = item.get(self.src_lang, None)
            target_text = item.get(self.tgt_lang, None)

            if source_text is None or target_text is None:
                idx += 1
                continue  # 해당 아이템 패스


            source_encoding = self.tokenizer(source_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            target_encoding = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

            return {
                'input_ids': source_encoding['input_ids'].flatten(),
                'attention_mask': source_encoding['attention_mask'].flatten(),
                'labels': target_encoding['input_ids'].flatten()
            }

