from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, 
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,BitsAndBytesConfig,
                          EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType
from src.data_preproceesing import TranslationDataset
from src.utils import get_target_modules
import wandb

wandb.login()

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
target_modules = get_target_modules(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=target_modules,
    task_type=TaskType.SEQ_2_SEQ_LM
)

target_modules = get_target_modules(model)

# QLoRA 적용된 모델 생성
model = get_peft_model(model, lora_config)

train_data_path = './data/train.json'
validation_dath_path = './data/validation.json'

# 데이터셋 로드
train_dataset = TranslationDataset(train_data_path, tokenizer)
val_dataset = TranslationDataset(validation_dath_path, tokenizer)


# 데이터 콜레이터 정의
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 파라미터 수 출력
print(f"Model number of parameters: {model.num_parameters()}")


# 파인튜닝 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb-200-distilled-600M_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=16,  # GPU 당 배치 크기
    gradient_accumulation_steps=1,  # 그래디언트 누적 단계
    warmup_ratio=0.1,
    learning_rate=5e-5,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
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
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,  # 연속적인 평가 단계 수 줄임
    early_stopping_threshold=0.001  # 개선 한도 낮춤
)

trainer = Seq2SeqTrainer(
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    model=model,
    callbacks=[early_stopping_callback]
)

trainer.train()
