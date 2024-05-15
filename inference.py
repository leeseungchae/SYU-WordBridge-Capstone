
"""
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 모델과 토크나이저를 불러옵니다.

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def translate(text, src_lang, tgt_lang):
    
    if src_lang == 'ko':
        
        tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-ko2en")
        model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-ko2en").to(device)
        
    else:
        tokenizer = AutoTokenizer.from_pretrained("NHNDQ/nllb-finetuned-en2ko")
        model = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-en2ko").to(device)
            
    # 입력 텍스트를 토큰화합니다.
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 모델을 사용하여 추론합니다.
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # 생성된 토큰을 디코딩하여 한국어 텍스트로 변환합니다.
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return translated_text

# 예제 문장
text = "이 코드는 영어 텍스트를 입력으로 받아서 한국어로 번역하는 과정을 설명합니다. translate_en_to_ko 함수는 다음 단계를 따릅니다:"
src_lang = 'ko'
tgt_lang = 'en'

# 번역 결과 출력
korean_translation = translate(text=text, src_lang=src_lang,tgt_lang=tgt_lang)
print(korean_translation)
