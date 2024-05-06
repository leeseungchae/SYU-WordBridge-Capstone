import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import wandb

wandb.login()

# 모델 및 토크나이저 로드
teacher_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B")
# model = AutoModelForSeq2SeqLM.from_pretrained('dhtocks/nllb-200-distilled-350M_en-ko', forced_bos_token_id=256098)
tokenizer = AutoTokenizer.from_pretrained('dhtocks/nllb-200-distilled-350M_en-ko', src_lang='eng_Latn', tgt_lang='kor_Hang')

# model = AutoModelForSeq2SeqLM.from_pretrained('dhtocks/nllb-200-distilled-350M_en-ko', load_in_4bit=True, device_map="auto", quantization_config={
#     "load_in_4bit": True,
#     "bnb_4bit_compute_dtype": "float16",  # 'float16' or 'bfloat16'
#     "bnb_4bit_use_double_quant": True,
#     "bnb_4bit_quant_type": "nf4"  # NormalFloat4
# })

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  # 'float16' or 'bfloat16'
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # NormalFloat4
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    'dhtocks/nllb-200-distilled-350M_en-ko',
    quantization_config=bnb_config,
    device_map="auto"
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

# Forced BOS 토큰 설정
teacher_model.config.forced_bos_token_id = 256098
model.config.forced_bos_token_id = 256098

# 파라미터 수 출력
print(f"Model number of parameters: {model.num_parameters()}")


import json
from datasets import Dataset

# JSON 파일 읽기
with open("data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 필요한 필드만 추출해 Dataset 객체로 변환
examples = [{"source_text": item["en"], "target_text": item["ko"]} for item in data["data"]]
dataset = Dataset.from_pandas(pd.DataFrame(examples))

# 데이터셋 분할 (예시: 90% 학습, 10% 평가)
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = [ex for ex in examples["source_text"]]
    targets = [ex for ex in examples["target_text"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
# 데이터셋에 전처리 함수 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 필요하지 않은 컬럼 제거
train_dataset = train_dataset.remove_columns(["source_text", "target_text"])
eval_dataset = eval_dataset.remove_columns(["source_text", "target_text"])


# 파인튜닝 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./nllb_350M",
    num_train_epochs=3,
    per_device_train_batch_size=24,  # GPU 당 배치 크기
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
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# 파인튜닝 실행
trainer.train()
