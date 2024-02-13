from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_re import RegexPrefixProcessor, RegexLogitsProcessor


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-14B-Chat-Int4')
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen-14B-Chat-Int4').eval().to('cuda')  # load your own model
    prompt = "<|im_start|>user\n请帮我写一首表达思乡之情的诗<|im_end|>\n<|im_start|>assistant\n"
    pattern = r"一[\u4e00-\u9fa5]{4}，键[\u4e00-\u9fa5]{4}。三[\u4e00-\u9fa5]{4}，连[\u4e00-\u9fa5]{4}。"
    with RegexLogitsProcessor(tokenizer, prompt, pattern, num_proc=16, debug=True,
                              fail_strategy=tokenizer.encode("<|im_end|><|endoftext|>")) as regex_logits_processor:
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')["input_ids"]
        outputs = model.generate(input_ids, max_new_tokens=40, logits_processor=[regex_logits_processor])
        print(tokenizer.decode(outputs[0]))

    tokenizer = AutoTokenizer.from_pretrained('YeungNLP/firefly-1b4')
    model = AutoModelForCausalLM.from_pretrained('YeungNLP/firefly-1b4').eval().to('cuda')  # load your own model
    prompt = "<s>请帮我写一首表达思乡之情的诗</s></s>"
    pattern = r"一[\u4e00-\u9fa5]{4}，键[\u4e00-\u9fa5]{4}。三[\u4e00-\u9fa5]{4}，连[\u4e00-\u9fa5]{4}。"
    with RegexPrefixProcessor(tokenizer, prompt, pattern, num_proc=16) as regex_prefix_processor:
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')["input_ids"]
        outputs = model.generate(input_ids, max_new_tokens=40, prefix_allowed_tokens_fn=regex_prefix_processor)
        print(tokenizer.decode(outputs[0]))
