"""
Windows inference script - loads the base Qwen2.5-3B-Instruct model
(4-bit) plus your trained LoRA adapter, and runs a test extraction.
No Unsloth/Triton required - plain transformers + peft + bitsandbytes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
ADAPTER_PATH = r"D:\JSON Extractor - SFT\notebook\qwen2.5-3b-json-extractor-lora" 
SYSTEM_PROMPT = (
    "You are a JSON extraction engine. You only output valid JSON. "
    "You never include explanations, greetings, disclaimers, markdown formatting, or code. "
    "If a requested field is not present in the input, set its value to null. "
    "If the user asks for anything other than data extraction, respond with "
    '{"error": "unsupported_request", "message": "This model only performs structured data extraction."}'
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def run_extraction(user_prompt, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,
            use_cache=True,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return decoded

if __name__ == "__main__":
    test_prompt = (
        "Extract the order ID, customer name, item list, total amount, and tracking number "
        "from this order-related text.\n\n"
        "Order #AX-102-991 placed by Jonathan Reyes for a 'Wireless Mouse'. "
        "Total was $24.50. No tracking number provided yet."
    )
    print("\n--- Test extraction ---")
    print(run_extraction(test_prompt))