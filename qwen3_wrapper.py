from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-8B"

tokeniser = AutoTokeniser.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "ehem..."
messages = [
    {"role": "user", "content": prompt}
]

text = tokeniser.apply_chat_template(
    messages,
    tokenise=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs - tokeniser([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

print(tokeniser.decode(output_ids, skip_special_tokens=True))