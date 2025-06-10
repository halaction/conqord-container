import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Union, Dict

from lm_polygraph.utils import WhiteboxModel, UEManager, Dataset
from lm_polygraph.estimators import Perplexity, MaximumSequenceProbability, LexicalSimilarity, NumSemSets
from lm_polygraph.ue_metrics import *
from lm_polygraph.ue_metrics.ue_metric import UEMetric

from .ece import ExpectedCalibrationError
from .verbal import VerbalUncertainty

class LMPolygraph:
    def __init__(self,
            model, 
            tokenizer, 
            model_path: str, 
            include: List[bool] = [True, True, True, True, True],
            lexical_similarity_metric: str='rougeL',
            verbal_confidence_prompt: str = None,
            verbal_confidence_regex: str = None
        ):
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        self.model = WhiteboxModel(model, tokenizer, model_path=model_path)
        self.estimators = []
        if include[0]:
            self.estimators.append(Perplexity())

        if include[1]:
            self.estimators.append(MaximumSequenceProbability())

        if include[2]:
            self.estimators.append(LexicalSimilarity(lexical_similarity_metric))
        
        if include[3]:
            self.estimators.append(NumSemSets())

        if include[4]:
            self.estimators.append(VerbalUncertainty(
                confidence_prompt = verbal_confidence_prompt,
                confidence_regex = verbal_confidence_regex
            ))

    def __call__(self,
            text_inputs: Union[str, List[str]],
            batch_size=None,
            verbose=False
        ) -> List[Dict]:
        """ 
        returns dict with:
           'generated_text': list of generated texts,
           'Perplexity': list of estimations,
           'MaximumSequenceProbability': list of estimations,
           'LexicalSimilarity': list of estimations,
           'NumSemSets': list of estimations,
           'VerbalUncertainty': list of estimations,
           'VerbalUncertainty_corrected': list of VerbalUncertainty estimations,
                with missing values set to 0, and clipped to between 0 and 1
        In all estimators higher values indicate more uncertain samples.
        """

        if type(text_inputs) is str:
            text_inputs = [text_inputs]

        if batch_size is None:
            batch_size = len(text_inputs)

        manager = UEManager(
            Dataset(text_inputs, [""] * len(text_inputs), batch_size=batch_size),
            self.model,
            self.estimators,
            [],
            [],
            [],
            ignore_exceptions=False,
            verbose=verbose,
        )

        with torch.no_grad():
            manager()
            
        results = {}
        for estimator in self.estimators:
            results[str(estimator)] = manager.estimations[estimator.level, str(estimator)]
            if str(estimator) == "VerbalUncertainty":
                results["VerbalUncertainty_corrected"] = np.clip(np.nan_to_num(results[str(estimator)], nan=0.0), 0.0, 1.0)

        results['generated_text'] = manager.stats.get("greedy_texts", None)

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
    def spearman_corr(cls, estimations: List[float], target: List[float]):
        return cls.metric(SpearmanRankCorrelation, estimations, target)

    @classmethod
    def kendall_corr(cls, estimations: List[float], target: List[float]):
        return cls.metric(KendallTauCorrelation, estimations, target)
