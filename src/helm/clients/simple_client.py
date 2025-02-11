from typing import List, Dict

from helm.common.cache import CacheConfig
from helm.common.request import wrap_request_time, Request, RequestResult, Sequence, Token
from helm.tokenizers.simple_tokenizer import SimpleTokenizer
from .client import CachingClient


class SimpleClient(CachingClient):
    """Implements some "models" that just generate silly things quickly just to debug the infrastructure."""

    def __init__(self, cache_config: CacheConfig):
        super().__init__(cache_config=cache_config)

    def make_request(self, request: Request) -> RequestResult:
        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "n": request.num_completions,
        }

        if request.model_engine in ["model1", "tutorial"]:

            def do_it():
                if request.model_engine == "model1":
                    return self.invoke_model1(raw_request)
                elif request.model_engine == "tutorial":
                    return self.invoke_model_tutorial(raw_request)

            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            completions = [
                Sequence(
                    text=text,
                    logprob=logprob,
                    tokens=[Token(text=text, logprob=logprob)],
                )
                for text, logprob in response["completions"].items()
            ]
        else:
            raise ValueError(f"Invalid model: {request.model}")

        return RequestResult(
            success=True,
            cached=False,
            request_time=0,
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def invoke_model1(self, raw_request: Dict) -> Dict:
        """
        Example: 7 2 4 6
        Completions (num_completions = 3):
        - 6
        - 4
        - 2
        """
        prompt_tokens: List[str] = SimpleTokenizer.tokenize_by_space(raw_request["prompt"])
        choices = reversed(prompt_tokens[-raw_request["n"] :])
        response = {"completions": dict((text, -i) for i, text in enumerate(choices))}
        return response

    def invoke_model_tutorial(self, raw_request: Dict) -> Dict:
        """Always returns: 'The model is generating some text. Hooray, the tutorial works! (Completion {i})'.
        This supports multiple completions.
        """
        response = {
            "completions": dict(
                (f"The model is generating some text. Hooray, the tutorial works! (Completion {i})", -i)
                for i in range(raw_request["n"])
            )
        }
        return response
