class LLMGatewayError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class LLMAuthenticationError(LLMGatewayError):
    pass


class LLMRateLimitError(LLMGatewayError):
    pass


class LLMTimeoutError(LLMGatewayError):
    pass


class LLMUpstreamError(LLMGatewayError):
    pass


class LLMRequestError(LLMGatewayError):
    pass


class LLMInvalidResponseError(LLMGatewayError):
    pass