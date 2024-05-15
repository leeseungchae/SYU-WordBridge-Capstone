import json
from torch.utils.data import Dataset
from src.utils import json_loads


class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, src_lang='ko', tgt_lang='en', max_length=128):
        
        self.data = json_loads(file_path)['data']
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item.get(self.src_lang, None)
        target_text = item.get(self.tgt_lang, None)

        while source_text is None or target_text is None:
            idx += 1
            if idx >= len(self.data):
                idx = 0  # 데이터셋의 끝에 도달한 경우 다시 처음부터 시작
            item = self.data[idx]
            source_text = item.get(self.src_lang, None)
            target_text = item.get(self.tgt_lang, None)

        # 한국어 입력 텍스트에 대한 토큰화 및 언어 코드 추가
        source_encoding = self.tokenizer(
            f"<2eng> {source_text}",
            max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        # 영어 출력 텍스트에 대한 토큰화
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )

        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

