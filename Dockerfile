FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN set -xe \
    && ls -a /etc/apt/sources.list.d/ \
    && rm /etc/apt/sources.list.d/cuda* \
    && apt-get update -y \
    && apt-get install -y python3-pip git

COPY requirements.txt /workspace

# Install python dependencies
RUN set -xe \
    && python3 -m pip install --no-cache-dir -r requirements.txt \
    && git clone https://github.com/halaction/conqord-container.git

# Patch lm_polygraph bug
RUN set -xe \
    && perl -i -0pe \
        "s/from transformers import \(.*?\)/# Patched import\nfrom transformers import AutoConfig, AutoTokenizer, BertForPreTraining, BertModel, RobertaModel, AlbertModel, AlbertForMaskedLM, RobertaForMaskedLM, get_linear_schedule_with_warmup\nfrom transformers.optimization import AdamW/s" \
        "/usr/local/lib/python3.10/dist-packages/lm_polygraph/generation_metrics/alignscore_utils.py"
