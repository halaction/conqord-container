import json
import numpy as np
import re
from typing import Dict

from lm_polygraph.estimators.estimator import Estimator
from lm_polygraph.estimators import Verbalized1S


def get_json_from_answer(response: str):
    """
    Extracts a JSON object from a string response, handling potential issues.

    Args:
        response (str): The string containing the potential JSON output.

    Returns:
        dict or None: The parsed JSON dictionary if successful, None otherwise.
    """
    full_json_match = re.search(r'\{.*\}', response, re.DOTALL)
    partial_json_match = re.search(r'\{.*', response, re.DOTALL)

    if full_json_match:
        json_string = full_json_match.group(0) 
    elif partial_json_match:
        json_string = partial_json_match.group(0) + '"}'
    else:
        return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} - Could not parse: {json_string}") 

    # Clean up common issues before parsing
    json_string = json_string.replace("'", '"') 
    json_string = re.sub(r',\s*([}\]])', r'\1', json_string) 

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} - Could not parse: {json_string}") 
    
    return None


standard_guess_regex = r"Guess[:\s]*(.*?)(?:\s*Probability|$)"
standard_prob_regex = r"Probability[:\s]*([-+]?\d*\.\d+)"

STANDARD_1S_CONFIDENCE_PROMPT = '''
System: You are a knowledgeable and very accurate chatbot.
Answer the provided question.
Output only one line with a valid JSON containing two fields: "confidence" - float from 0.0 to 1.0 and "answer" - str with a correct answer.

-----Example-----
Example Question:
Who is the author of The Jungle Book?

Example Output:
{{"confidence": 0.98, "answer": "Rudyard Kipling"}}

-----End of Example-----

$Question: 
${question}

$Output:
'''

class Verbalized1S_mod(Estimator):
    def __init__(self, prompt="", confidence_regex="", guess_regex=""):
        self.prompt = prompt
        super().__init__(["greedy_texts"], "sequence")
    
    def get_prompt(self):
        return self.prompt

    def __str__(self):        
        return f"Verbalized1S"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        for answer in stats["greedy_texts"]:
            print(f'Verbalized1S_mod {answer=}')
            try:
                json_answer = get_json_from_answer(answer)
                print(f'Verbalized1S_mod {json_answer=}')
                confidence = json_answer['confidence']
                guess = json_answer['answer']
                ue = 1 - confidence
            except Exception as e:
                ue = -1
                guess = ""

            ues.append((ue, guess))

        return ues

STANDARD_2S_CONFIDENCE_PROMPT = '''
System: You are a helpful chatbot. For each pair of $Question and $Answer, please output the probability that the $Answer is correct for the $Question. Just output the probability (0.0 to 1.0).

$Question: {0}

$Answer: {1}

$Probability:
'''

class Verbalized2S_mod(Estimator):
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
    
topk_guess_regex = r"G\d+:\s*(.+)"
topk_prob_regex = r"P\d+:\s*([-+]?\d*\.\d+)"

# from https://arxiv.org/pdf/2305.14975
TOPK_CONFIDENCE_PROMPT = '''
Provide your ${k} best guesses and the probability that each is correct (0.0 to
1.0) for the following question. Give ONLY the guesses and probabilities, no other
words or explanation. For example:\n\nG1: <first most likely guess, as short as
possible; not a complete sentence, just the guess!>\n\nP1: <the probability between
0.0 and 1.0 that G1 is correct, without any extra commentary whatsoever; just
the probability!> ... G${k}: <${k}-th most likely guess, as short as possible;
not a complete sentence, just the guess!>\n\nP${k}: <the probability between 0.0
and 1.0 that G${k} is correct, without any extra commentary whatsoever; just the
probability!> \n\nThe question is: ${question}
'''

class Verbalized1STopK(Estimator):
    def __init__(self, k=4, prompt="", confidence_regex="", guess_regex=""):
        self.k = k
        self.prompt = prompt
        self.confidence_regex = confidence_regex
        self.guess_regex = guess_regex
        super().__init__(["greedy_texts"], "sequence")

    def get_prompt(self):
        return self.prompt

    def __str__(self):
        return f"Verbalized1STop{self.k}"
    
    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        ues = []
        guesses = []
        conf_re = re.compile(self.confidence_regex)
        guess_re = re.compile(self.guess_regex)
        for answer in stats["greedy_texts"]:
            guess = re.findall(guess_re, answer)
            prob = re.findall(conf_re, answer)

            try:
                assert len(guess) == len(prob)
                prob = [float(p) for p in prob]
                max_index = prob.index(max(prob))

                ue = 1 - float(prob[max_index])
                guess = guess[max_index]

            except:
                ue = np.nan
                guess = ""
            
            ues.append((ue, guess))

        return ues