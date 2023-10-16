from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_re import RegexPrefixProcessor


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('YeungNLP/firefly-1b4')
    model = AutoModelForCausalLM.from_pretrained('YeungNLP/firefly-1b4').eval().to('cuda')  # load your own model
    prompt = "<s>请帮我写一首表达思乡之情的诗</s></s>"
    pattern = r"一[\u4e00-\u9fa5]{4}，键[\u4e00-\u9fa5]{4}。三[\u4e00-\u9fa5]{4}，连[\u4e00-\u9fa5]{4}。"
    with RegexPrefixProcessor(tokenizer, prompt, pattern, num_proc=16) as regex_prefix_processor:
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')["input_ids"]
        outputs = model.generate(input_ids, max_new_tokens=40, prefix_allowed_tokens_fn=regex_prefix_processor)
        print(tokenizer.decode(outputs[0]))
