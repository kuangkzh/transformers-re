import warnings
import regex
import multiprocess
import torch


class RegexPrefixProcessor:
    """
    A regex prefix constraint for transformers.
    Usage example:
    >>> with RegexPrefixProcessor(tokenizer, prompt, pattern) as regex_prefix_processor:
    >>>   model.generate(prefix_allowed_tokens_fn=regex_prefix_processor)
    Attributes:
        generated_text(str): Cached text. Generation can be restored from it to prevent running error of regex mismatch
        match(regex.Match): The Match object of generated text and pattern
    """
    def __init__(self, tokenizer, prompt, pattern, num_proc=1, fail_strategy='eos', debug=False):
        """
        :param tokenizer: transformers.Tokenizer
        :param prompt: the prompt input into model.
        :param pattern: regex pattern.
        :param num_proc: the number of processors to process regex match.
        :param fail_strategy: default: 'eos'. The strategy when no token can use. It can be:
                            - List[int]: token ids when no token can use.
                            - 'eos': automatically choose the tokenizer.eos_token_id
        :param debug: default: False. Control the debug information output.
        """
        self.tokenizer = tokenizer
        self.pattern = regex.compile(pattern)
        self.prompt_token_len = len(tokenizer(prompt)['input_ids'])
        self.fail_strategy = [tokenizer.eos_token_id] if fail_strategy == 'eos' else fail_strategy
        self.debug = debug
        vocab = [(idx, tokenizer.decode(idx)) for _, idx in tokenizer.get_vocab().items()]

        size = len(vocab) + (-len(vocab) % num_proc)  # find the minimum size to be divided by n
        self.q_in = [multiprocess.Queue() for _ in range(num_proc)]
        self.q_out = [multiprocess.Queue() for _ in range(num_proc)]

        def regex_service(i):
            vl = vocab[size // num_proc * i: size // num_proc * (i + 1)]

            def _regex_match():
                while True:
                    s = self.q_in[i].get()
                    self.q_out[i].put([idx for idx, ch in vl if self.pattern.fullmatch(s + ch, partial=True)])

            return _regex_match

        self.process_list = [multiprocess.Process(target=regex_service(i), daemon=True) for i in range(num_proc)]

        self.generated_text = ""
        self.match = None

    def __enter__(self):
        [proc.start() for proc in self.process_list]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        [proc.kill() for proc in self.process_list]
        [q.close() for q in self.q_in]
        [q.close() for q in self.q_out]
        [proc.join() for proc in self.process_list]
        [proc.close() for proc in self.process_list]
        if exc_type is not None:
            warnings.warn("An Exception happened during text generation. It happens usually when all logits are too "
                          "small to be NaN, thus no candidates token can be choosen from\n"
                          "Set debug=True to track the generation process. \n"
                          "You can restore your text from RegexPrefixProcessor.generated_text.")

    def __call__(self, batch_id, input_ids):
        self.generated_text = self.tokenizer.decode(input_ids[self.prompt_token_len:])

        [q.put(self.generated_text) for q in self.q_in]
        self.match = self.pattern.match(self.generated_text, partial=True)
        candidates = sum([q.get() for q in self.q_out], [])

        # candidates = [idx for idx, ch in vocab if self.pattern.fullmatch(generated_text+ch, partial=True)]
        candidates = candidates if candidates else self.fail_strategy
        if self.debug:
            candidates_repr = (f"{len(candidates)} tokens" if len(candidates) > 10 else candidates)
            print(f"generated:{self.generated_text} candidates:{candidates_repr}")
        return candidates


def get_token_mapping(tokenizer, s):
    token_ids = tokenizer.encode(s)
    if tokenizer.is_fast:
        return [tokenizer(s).token_to_chars(i) for i in range(len(token_ids))]
    token_ids = tokenizer.encode(s)
    pos, token2span = 0, []
    for i in range(len(token_ids)):
        token = regex.escape(tokenizer.decode(token_ids[i]))
        token2span.append(regex.search(token, s, pos=pos))
        assert token2span[-1].start() == pos, "Some characters are skipped without token related."
        pos = token2span[-1].end()
    return token2span


def getter(x):
    return x() if callable(x) else x


def extract(text_span, s, token2span, strategy="expand"):
    start_dict = {0: 0, len(s): len(token2span)}
    start_dict.update({getter(sp.start): i for i, sp in enumerate(token2span) if sp is not None})
    end_dict = {0: -len(token2span)-1, len(s): len(token2span)}
    end_dict.update({getter(sp.end): i+1 for i, sp in enumerate(token2span) if sp is not None})

    start, end = text_span[0], text_span[1]
    if start in start_dict and end in end_dict:
        return start, end, start_dict[start], end_dict[end]

    if strategy == "expand":
        start = max([x for x in start_dict if x <= start]+[0])
        end = min([x for x in end_dict if x >= end]+[len(s)])
        return start, end, start_dict[start], end_dict[end]
    elif strategy == "shrink":
        start = min([x for x in start_dict if x >= start]+[len(s)])
        end = max([x for x in end_dict if x <= end]+[0])
        return start, end, start_dict[start], end_dict[end]
    elif strategy == "error":
        raise RuntimeError("The match span contains incomplete tokens")
    raise RuntimeError("Invalid strategy. Should be expand, shrink or error")


class TokenizedPattern:
    """
    A regex pattern compiled with a tokenizer and applies to token or tensors.
    """

    def __init__(self, tokenizer, pattern, strategy="expand", token_mapping_func=get_token_mapping):
        """
        Args:
            tokenizer: The using tokenizer
            pattern: A regex pattern to match strings
            strategy: Since the match string may cut off some tokens. It decides what to do when truncation happends.
                    - 'expand': expand the match span to the minimum token span covers the string span.
                    - 'shrink': shrink the match span to the maximum token span be covered by the string span.
                    - 'error': do nothing but raise an error.
            token_mapping_func: A function where f(tokenizer, str) -> List[Span]. The list length should equals to the
                                token length, in which each span corresponding to its token's character span in string.
        """
        self.tokenizer = tokenizer
        self.pattern = regex.compile(pattern)
        assert strategy in ("expand", "shrink", "error")
        self.strategy = strategy
        self.token_mapping_func = token_mapping_func

    def match(self, string, pos=None, endpos=None):
        match_obj = self.pattern.match(string, pos, endpos)
        return None if match_obj is None else TokenizedMatch(
            self.tokenizer,
            match_obj,
            strategy=self.strategy,
            token_mapping_func=self.token_mapping_func
        )

    def search(self, string, pos=None, endpos=None):
        match_obj = self.pattern.search(string, pos, endpos)
        return None if match_obj is None else TokenizedMatch(
            self.tokenizer,
            match_obj,
            strategy=self.strategy,
            token_mapping_func=self.token_mapping_func
        )

    def finditer(self, string, pos=None, endpos=None):
        pass


class TokenizedMatch:
    """
    A regex match result corresponding to a TokenizedPattern and a string
    """
    def __init__(self, tokenizer, match_obj, strategy="expand", token_mapping_func=get_token_mapping):
        self.tokenizer = tokenizer
        self.match_obj = match_obj
        assert strategy in ("expand", "shrink", "error")
        self.strategy = strategy
        self.token_mapping_func = token_mapping_func
        self.string = match_obj.string
        self.token2span = self.token_mapping_func(self.tokenizer, self.string)

    def __len__(self):
        return len(self.match_obj)

    def span(self, index=0, of_token=False):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        return (x, y) if of_token else (a, b)

    def start(self, index=0, of_token=False):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        return x if of_token else a

    def end(self, index=0, of_token=False):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        return y if of_token else b

    def group(self, index=0, of_token=False):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        if of_token:
            return self.tokenizer(self.string).input_ids[x:y]
        return self.string[a:b]

    def mask(self, index=0):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        mask = torch.zeros_like(self.tokenizer(self.string, return_tensors="pt").input_ids[0])
        mask[x:y] = 1
        return mask

    def masked_select(self, tensor, index=0, dim=-1):
        span = self.match_obj.span(index)
        a, b, x, y = extract(span, self.string, self.token2span, self.strategy)
        mask = torch.arange(x, y, dtype=torch.int64, device=tensor.device)
        return tensor.index_select(dim=dim, index=mask)
