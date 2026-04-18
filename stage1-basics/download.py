from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"


def download_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   #加载的什么
    model = AutoModelForCausalLM.from_pretrained(  #加载模型？
        MODEL_NAME,
        dtype=torch.bfloat16,   #精度
        device_map="auto",     #什么意思
    )
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "你是什么版本的模型"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs_cached = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,   # 默认就是 True
            do_sample=False,  # greedy 确保可对比
        )
        generated_cached = tokenizer.decode(
            outputs_cached[0], skip_special_tokens=True
        )
        print(generated_cached)

    
    
download_model()