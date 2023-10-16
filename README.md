# Transformers-Re
A Regular Expression constraint for Language Models of transformers. With this module, you can force the LLMs to 
generate following your regex. Using regex in tokens and tensors are also implemented in this project.

本项目支持通过正则表达式控制LLMs输出，同时还实现了通过正则表达式抽取token或tensor。


## Installation
```shell script
pip install transformers-re
```


### RegexPrefixProcessor

A regex prefix constraint for transformers.

**\_\_init\_\_**:
- tokenizer: transformers.Tokenizer
- prompt: the prompt input into model.
- pattern: regex pattern.
- num_proc: the number of processors to process regex match.
- fail_strategy: default: 'eos'. The strategy when no token can use. It can be:
    - List[int]: token ids when no token can use.
    - 'eos': automatically choose the tokenizer.eos_token_id
- debug: default: False. Control the debug information output.


**Attributes**:
- generated_text(str): Cached text. Generation can be restored from it to prevent running error of regex mismatch
- match(regex.Match): The Match object of generated text and pattern

**Usage example**:

```python
>>> with RegexPrefixProcessor(tokenizer, prompt, pattern) as regex_prefix_processor:
>>>   model.generate(prefix_allowed_tokens_fn=regex_prefix_processor)
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers_re import RegexPrefixProcessor

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('YeungNLP/firefly-1b4')
    model = AutoModelForCausalLM.from_pretrained('YeungNLP/firefly-1b4').eval().to('cuda')  # load your own model
    prompt = "<s>请帮我写一首表达思乡之情的诗</s></s>"
    pattern = r"一[\u4e00-\u9fa5]{4}，键[\u4e00-\u9fa5]{4}。三[\u4e00-\u9fa5]{4}，连[\u4e00-\u9fa5]{4}。"
    with RegexPrefixProcessor(tokenizer, prompt, pattern, num_proc=16) as regex_prefix_processor:
        input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')["input_ids"]
        outputs = model.generate(input_ids, max_new_tokens=20, prefix_allowed_tokens_fn=regex_prefix_processor)
        print(tokenizer.decode(outputs[0]))

# <s>请帮我写一首表达思乡之情的诗</s></s>一叶落叶归，键盘敲击声。三更梦回乡，连日思乡情。</s>
```


### TokenizedPattern

A regex pattern compiled with a tokenizer and applies to token or tensors.

**\_\_init\_\_**:

- tokenizer: The using tokenizer
- pattern: A regex pattern to match strings
- strategy: Since the match string may cut off some tokens. It decides what to do when truncation happends.
    - 'expand': expand the match span to the minimum token span covers the string span.
    - 'shrink': shrink the match span to the maximum token span be covered by the string span.
    - 'error': do nothing but raise an error.
- token_mapping_func: A function where f(tokenizer, str) -> List[Span]. The list length should equals to the
                      token length, in which each span corresponding to its token's character span in string.

**Methods**:
- match(self, string, pos=None, endpos=None) -> TokenizedMatch
- search(self, string, pos=None, endpos=None) -> TokenizedMatch


### TokenizedMatch
The regex match result corresponding to a TokenizedPattern and a string.

**Methods**:
- span(self, index=0, of_token=False)
- start(self, index=0, of_token=False)
- end(self, index=0, of_token=False)
- group(self, index=0, of_token=False)
- mask(self, index=0)
- masked_select(self, tensor, index=0, dim=-1)

**Usage example**:

```python
s = "This is an example text"
token_ids = tokenizer(s, return_tensors="pt").input_ids
tokens = [tokenizer.decode(t) for t in tokenizer(s).input_ids]
print(token_ids, tokens, sep="\n")

"""
tensor([[3180,  579,  593, 3392, 2895]])
['This', ' is', ' an', ' example', ' text']
"""
```

```python
from transformers_re import TokenizedPattern

a, b = TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).span() # Get the text span, expanded according to token
print(s[a:b])
# " example text"

a, b = TokenizedPattern(tokenizer, "ample(.*)", "shrink").search(s).span() # Strategy shrink
print(s[a:b])
# " text"

a, b = TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).span(of_token=True) # Get token span
print(token_ids[0, a:b])
# tensor([3392, 2895])

a, b = TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).span(index=1, of_token=True) # Select group 1 by index
print(token_ids[0, a:b])
# tensor([2895])

mask = TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).mask() # Get mask tensor
print(mask)
# tensor([0, 0, 0, 1, 1])
```

```python
import torch
torch.manual_seed(42)
x = torch.randint(0, 10, (5, 5))
"""
tensor([[2, 7, 6, 4, 6],
        [5, 0, 4, 0, 3],
        [8, 4, 0, 4, 1],
        [2, 5, 5, 7, 6],
        [9, 6, 3, 1, 9]])
"""


print(TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).masked_select(x, dim=0))
"""
tensor([[2, 5, 5, 7, 6],
        [9, 6, 3, 1, 9]])
"""


print(TokenizedPattern(tokenizer, "ample(.*)", "expand").search(s).masked_select(x, dim=1))
"""
tensor([[4, 6],
        [0, 3],
        [4, 1],
        [7, 6],
        [1, 9]])
"""
```

## Limitation
THe main bottleneck of regex constraint generation is the match processing of regex.
Although we can use multiprocess in this project to accelerate the procedure, the 
time consumption proportion of regex match is still too high.

The time cost can be reduced more if incremental matching of regex or some GPU regex 
engine can be implement in this project.
