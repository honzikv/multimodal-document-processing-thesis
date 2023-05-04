from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from .configuration_layoutlmv3 import LayoutLMv3GithubConfig
from .modeling_layoutlmv3 import (
    LayoutLMv3GithubForTokenClassification,
    LayoutLMv3GithubForQuestionAnswering,
    LayoutLMv3GithubForSequenceClassification,
    LayoutLMv3GithubModel,
)
from .tokenization_layoutlmv3 import LayoutLMv3GithubTokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3GithubTokenizerFast


AutoConfig.register("layoutlmv3_github", LayoutLMv3GithubConfig)
AutoModel.register(LayoutLMv3GithubConfig, LayoutLMv3GithubModel)
AutoModelForTokenClassification.register(LayoutLMv3GithubConfig, LayoutLMv3GithubForTokenClassification)
AutoModelForQuestionAnswering.register(LayoutLMv3GithubConfig, LayoutLMv3GithubForQuestionAnswering)
AutoModelForSequenceClassification.register(LayoutLMv3GithubConfig, LayoutLMv3GithubForSequenceClassification)
AutoTokenizer.register(
    LayoutLMv3GithubConfig, slow_tokenizer_class=LayoutLMv3GithubTokenizer, fast_tokenizer_class=LayoutLMv3GithubTokenizerFast
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
