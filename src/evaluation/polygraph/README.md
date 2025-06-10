# Benchmarking with LM-Polygraph

## How-to

Calculates uncertainty estimations, listed below, and generated texts.

`model` and `tokenizer` are HuggingFace instances, `model_path` is the same as `pretrained_model_name_or_path` in HuggingFace `from_pretrained`.

```
from polygraph import LMPolygraph

polygraph = LMPolygraph(model, tokenizer, model_path)
results = polygraph(questions, batch_size=batch_size, verbose=True)
```

## Uncertainty Estimation (UE)

> Uncertainty is a fundamental concept in machine learning and statistics, indicating that model predictions have a degree of variability due to the lack of complete information (Vashurin, et al., 2024).

Language model probability formulation:
$$
p(y | x, \theta) = \prod_{l}^{L}{p(y_{l} | y_{<l}, x, \theta)}
$$

Classification of UE methods by access mode:
- White-box - requires inputs, outputs and parameters, may be represented as $ U(y, x, \theta) $
- Black-box - requires only inputs and outputs, may be represented as $ U(y, x) $ or $ U(y) $

Classification of UE methods by approach:
- Information-based - utilizes model probability
- Sample diversity - uses an arbitrary similarity measure over multiple model samples
- Reflexive - requires another LM to produce uncertainty scores

### Perplexity

Simplistic white-box information-based method.

$$
U_{P}(y, x, \theta) = exp(-\frac{1}{L}\log{p(y | x, \theta)})
$$

### Mean Sequence Probability (MSP)

Simplistic white-box information-based method.

$$
U_{MSP}(y, x, \theta) = 1 - p(y | x, \theta)
$$

### Lexical Similarity

Black-box method based on sample diversity. Defined for an any lexical similarity measure $ s(y_i, y_j) $, such as ROUGE-N, ROUGE-L, and BLEU. In our use case it is ROUGE-L. LexSim is computed as average similarity between model outputs over all distinct sample pairs.

$$
U_{LexSim}(y_1, \ldots, y_K) = \frac{2}{K(K-1)} \sum_{i}^{K} \sum_{j > i}^{K} s(y_i, y_j)
$$

### Number of Semantic Sets (NumSet)

Black-box method based on sample diversity. NumSet is computed as number of distinct sample pairs that satisfy the following condition: probability of entailment between model outputs is higher than probability of contradiction. Probabilities are produced by an underlying NLI-instructed model.

$$
U_{NumSet}(y_1, \ldots, y_K) = \sum_{i}^{K} \sum_{j > i}^{K} \mathbb{I}\{p_{entails}(y_i, y_j) > p_{contradicts}(y_i, y_j)\} \cdot \mathbb{I}\{p_{entails}(y_j, y_i) > p_{contradicts}(y_j, y_i)\}
$$

### Verbalized Uncertainty (VU)

Black-box reflexive method. VU utilizes a separate instruction-tuned LLM to produce uncertainty score for primary model output. Includes N-Shot and Top-K varieties. Approaches such as CONQORD and Self-Evaluation are a subset of VU.

We use the following prompt, borrowed from CONQORD:

```
System: You are a helpful chatbot. For each pair of $Question and $Answer, please output the probability that the $Answer is correct for the $Question. Just output the probability (0.0 to 1.0).

$Question: {0}

$Answer: {1}

$Probability:
```

## Metrics

```
from polygraph import LMPolygraph

LMPolygraph.prr(estimations, target)
LMPolygraph.ece(estimations, target)
LMPolygraph.spearman_corr(estimations, target)
LMPolygraph.kendall_corr(estimations, target)
```

In `estimations` higher values indicate more uncertainty.
In `target` higher values indicate less uncertainty.

Metrics are used to benchmark alignment between quality and uncertainty scores.

### Prediction Rejection Ratio (PRR)

PRR is based on Prediction Rejection (PR) curve. PR indicates how the average quality $ Q(f(x_i), y_i) $ of the instances with uncertainty $ U(x_i) < u $ depends on the value of the rejection parameter $ u $. Similar to ROC-AUC, PR-AUC measures how well-ordered quality-uncertainty set is - higher values of PR-AUC indicate that lower uncertainty is associated with higher quality, and vice versa. PR-AUC produced by $ U(x_i) $ denoted as $ PR-AUC_U $ is then standardized by worst and best PR-AUC, which correspond to random UE $ PR-AUC_R $ and ideally aligned UE $ PR-AUC_O $.

$$
PRR = \frac{PR-AUC_U - PR-AUC_R}{PR-AUC_O - PR-AUC_R}
$$

### Expected Calibration Error (ECE)

$$
ECE = \sum_{i}^{M}{\frac{|B_i|}{N}{|Q_{B_i}(f(x), y) - C_{B_i}(x)|}}
$$

### Spearman Correlation

### Kendall Correlation



## References

...