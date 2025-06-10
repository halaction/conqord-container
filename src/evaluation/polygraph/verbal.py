import numpy as np
import re
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator

STANDARD_CONFIDENCE_PROMPT = '''
System: You are a helpful chatbot. For each pair of $Question and $Answer, please output the probability that the $Answer is correct for the $Question. Just output the probability (0.0 to 1.0).

$Question: {0}

$Answer: {1}

$Probability:
'''

class VerbalUncertainty(Estimator):
    def __init__(self, confidence_prompt: str = None,
                 confidence_regex: str = None,
                 max_new_tokens: int = 10):
        if confidence_prompt is None:
            confidence_prompt = STANDARD_CONFIDENCE_PROMPT
        
        if confidence_regex is None:
            confidence_regex = r"(\d+\.?\d+)"

        self.confidence_prompt = confidence_prompt
        self.confidence_regex = confidence_regex
        self.max_new_tokens = max_new_tokens
        super().__init__(["input_texts", "greedy_texts"], "sequence")

    def __str__(self):
        return "VerbalUncertainty"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        model = stats["model"]

        chats = []
        prompts = stats["input_texts"]
        guesses = stats["greedy_texts"]
        for prompt, guess in zip(prompts, guesses):
            chats.append(self.confidence_prompt.format(prompt, guess))

        out = model.generate_texts(
            chats,
            min_new_tokens=1,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
        )

        ues = []
        conf_re = re.compile(self.confidence_regex)
        for answer in out:
            match = re.search(conf_re, answer)

            try:
                ue = 1 - float(match.groups()[0])
            except AttributeError:
                ue = np.nan

            ues.append(ue)

        return np.array(ues)
