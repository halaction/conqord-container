import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union, Dict, Any

from lm_polygraph.utils import WhiteboxModel, UEManager, Dataset
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability, LexicalSimilarity, NumSemSets, Verbalized1S
from lm_polygraph.ue_metrics import *
from lm_polygraph.ue_metrics.ue_metric import UEMetric

from .ece import ExpectedCalibrationError
from .verbal import *


BASE_PROMPT = '''
{question}
'''


class LMPolygraph:
    default_estimator_config = {
        'Perplexity': {},
        'MaximumSequenceProbability': {},
        'LexicalSimilarity': {'metric': 'rougeL'},
        'NumSemSets': {},
        'VerbalUncertainty': {
            'confidence_regex': standard_prob_regex,
            'guess_regex': standard_guess_regex,
            'prompt': STANDARD_1S_CONFIDENCE_PROMPT
        },
        'VerbalUncertaintyTopK': {
            'k': 4,
            'confidence_regex': topk_prob_regex ,
            'guess_regex': topk_guess_regex ,
            'prompt': TOPK_CONFIDENCE_PROMPT
        }
    }
    estimator_classes = {
        'Perplexity': Perplexity,
        'MaximumSequenceProbability': MaximumSequenceProbability,
        'LexicalSimilarity': LexicalSimilarity,
        'NumSemSets': NumSemSets,
        'VerbalUncertainty': Verbalized1S_mod,
        'VerbalUncertaintyTopK': Verbalized1STopK,
    }

    def __init__(
        self,
        model,
        tokenizer,
        model_path: str,
        estimator_config: Union[None, List[str], Dict[str, Dict]] = None
    ):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.model = WhiteboxModel(model, tokenizer, model_path=model_path)

        # 1) If no config, use all estimators with defaults
        if estimator_config is None:
            self.estimator_config = self.default_estimator_config
        # 2) If it's a list of estimator names, each uses default config
        elif isinstance(estimator_config, list):
            self.estimator_config = {
                name: self.default_estimator_config[name]
                for name in estimator_config
            }
        # 3) If it's a dict {est_name: config_dict}, 
        #    all the necessary hyperparameters must be provided
        elif isinstance(estimator_config, dict):
            self.estimator_config = estimator_config
        else:
            raise ValueError("estimator_config must be None, list, or dict")

        self.estimators = {}
        for est_name, est_conf in self.estimator_config.items():
            if est_name not in self.estimator_classes:
                raise ValueError(f"Unknown estimator: {est_name}")

            EstimatorCls = self.estimator_classes[est_name]
            instance = EstimatorCls(**est_conf)
            self.estimators[est_name] = instance

    def _run_man(self, text_inputs, estimators, batch_size, verbose):
        man = UEManager(
            Dataset(text_inputs, [""] * len(text_inputs), batch_size=batch_size),
            self.model, estimators,
            [], [], [], ignore_exceptions=False, verbose=verbose
        )
        with torch.no_grad():
            man()

        generated_text = man.stats.get("greedy_texts", None)
        return generated_text, man.estimations

    def _vu_extract(self, est, estimations):
        values, guesses = list(zip(*estimations[(est.level, str(est))]))
        return guesses, np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)

    def __call__(self,
            text_inputs: Union[str, List[str]],
            batch_size=None,
            verbose=False
        ) -> Dict:
        """ 
        Runs all included estimators.
        Each type of verbalized uncertainty that requires separate prompting runs separately.
        Returns dict with:
           'generated_text': list of generated texts,
           'Perplexity': list of estimations,
           'MaximumSequenceProbability': list of estimations,
           'LexicalSimilarity': list of estimations,
           'NumSemSets': list of estimations,
           'vu_generated_text': generated text in 1S verbal uncertainty,
           'Verbalized1S': list of estimations
        In all estimators higher values indicate more uncertain samples.
        """

        if type(text_inputs) is str:
            text_inputs = [text_inputs]
        if batch_size is None:
            batch_size = len(text_inputs)

        results: Dict[str, List] = {}

        # STEP 1: Estimators that don't need a special prompt
        plain_estimators = []
        for name, est in self.estimators.items():
            # Verbalized Uncertainty is handled separately
            if name not in ("VerbalUncertainty", "VerbalUncertaintyTopK"):
                plain_estimators.append(est)

        if plain_estimators:
            prompt_inp = [
                BASE_PROMPT.format(question=t) for t in text_inputs
            ]
            generated_text, estimations = self._run_man(prompt_inp, plain_estimators, batch_size, verbose)

            results['generated_text'] = generated_text
            for est in plain_estimators:
                results[str(est)] = estimations[(est.level, str(est))]
        
        # STEP 2: Prompt-based estimators
        if "VerbalUncertainty" in self.estimators:
            est = self.estimators["VerbalUncertainty"]
            prompt_inp = [
                est.get_prompt().format(question=t) for t in text_inputs
            ]
            generated_text, estimations = self._run_man(prompt_inp, [est], batch_size, verbose)

            results['vu_generated_text'] = generated_text
            results['vu_extracted_guess'], results[str(est)] = self._vu_extract(est, estimations)
        
        if "VerbalUncertaintyTopK" in self.estimators:
            est = self.estimators["VerbalUncertaintyTopK"]
            prompt_inp = [
                est.get_prompt().format(k=est.k, question=t) for t in text_inputs
            ]
            generated_text, estimations = self._run_man(prompt_inp, [est], batch_size, verbose)

            results['vu_topk_generated_text'] = generated_text
            results['vu_topk_extracted_guess'], results[str(est)] = self._vu_extract(est, estimations)
        
        return results

    @staticmethod
    def metric(
            metric_class: UEMetric,
            estimations: List[float],
            target: List[float],
            *args
        ):
        """
        Parameters:
            estimator (List[int]): a batch of uncertainty estimations.
                Higher values indicate more uncertainty.
            target (List[int]): a batch of ground-truth uncertainty estimations.
                Higher values indicate less uncertainty.
        Returns:
            float: metric value
                Higher values indicate better uncertainty estimations.
        """
        metric = metric_class(*args)
        return metric(estimations, target)

    @classmethod
    def prr(cls, estimations: List[float], target: List[float], max_rejection: float = 1.0):
        """
        Parameters:
            max_rejection (float): a maximum proportion of instances that will be rejected.
                1.0 indicates entire set (PRR per se), 0.5 - half of the set
        """
        return cls.metric(PredictionRejectionArea, estimations, target, max_rejection)

    @classmethod
    def ece(cls, estimations: List[float], target: List[float]):
        return cls.metric(ExpectedCalibrationError, estimations, target)

    @classmethod
    def rocauc(cls, estimations: List[float], target: List[float]):
        return cls.metric(ROCAUC, estimations, [round(i) for i in target])

    @classmethod
    def spearman_corr(cls, estimations: List[float], target: List[float]):
        return cls.metric(SpearmanRankCorrelation, estimations, target)

    @classmethod
    def kendall_corr(cls, estimations: List[float], target: List[float]):
        return cls.metric(KendallTauCorrelation, estimations, target)

    @classmethod
    def list_metrics(cls):
        return [
            cls.prr,
            cls.ece,
            cls.rocauc,
            cls.spearman_corr,
            cls.kendall_corr
        ]