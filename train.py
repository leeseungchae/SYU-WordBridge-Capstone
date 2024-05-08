import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, 
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

wandb.login()

# 데이터셋 클래스 정의
class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, src_lang='en', tgt_lang='ko', max_length=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']
        
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text = item['en']
        target_text = item['ko']

        source_encoding = self.tokenizer(source_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        target_encoding = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')

        return {
            'input_ids': source_encoding['input_ids'].flatten(),
            'attention_mask': source_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

# NLLB 모델 및 토크나이저 로드
model_name = 'facebook/nllb-200-distilled-600M'
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang='eng_Latn', tgt_lang='kor_Hang')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  # 'float16' or 'bfloat16'
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # NormalFloat4
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    'facebook/nllb-200-distilled-600M',
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=[
        "encoder.layers.0.self_attn.k_proj",
        "encoder.layers.0.self_attn.v_proj",
        "encoder.layers.0.self_attn.q_proj",
        "encoder.layers.0.self_attn.out_proj",
        "encoder.layers.0.encoder_attn.k_proj",
        "encoder.layers.0.encoder_attn.v_proj",
        "encoder.layers.0.encoder_attn.q_proj",
        "encoder.layers.0.encoder_attn.out_proj",
        "decoder.layers.0.self_attn.k_proj",
        "decoder.layers.0.self_attn.v_proj",
        "decoder.layers.0.self_attn.q_proj",
        "decoder.layers.0.self_attn.out_proj",
        "decoder.layers.0.encoder_attn.k_proj",
        "decoder.layers.0.encoder_attn.v_proj",
        "decoder.layers.0.encoder_attn.q_proj",
        "decoder.layers.0.encoder_attn.out_proj",
    ],
    task_type=TaskType.SEQ_2_SEQ_LM
)


# QLoRA 적용된 모델 생성
model = get_peft_model(model, lora_config)

# 데이터셋 로드
train_dataset = TranslationDataset('./data/train.json', tokenizer)
val_dataset = TranslationDataset('./data/validation.json', tokenizer)


# 데이터 콜레이터 정의
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 파라미터 수 출력
print(f"Model number of parameters: {model.num_parameters()}")


# 파인튜닝 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-200-distilled-600M_finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=16,  # GPU 당 배치 크기
    gradient_accumulation_steps=2,  # 그래디언트 누적 단계
    warmup_ratio=0.1,
    learning_rate=5e-5,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    max_grad_norm=1.0,
    save_steps=1000,
    fp16=True,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    save_total_limit=3,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False,
    group_by_length=True,
    report_to="wandb",
    run_name="nllb_finetuning_run"
)
# 트레이너 설정
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 파인튜닝 실행
trainer.train()
