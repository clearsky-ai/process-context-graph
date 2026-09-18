"""Shared Azure OpenAI client factory using keyless (Microsoft Entra ID) auth.

Authentication is done with a bearer token from ``azure-identity`` instead of an
API key. When ``AZURE_OPENAI_CLIENT_ID`` is set, that user-assigned managed
identity / app registration is preferred; otherwise the full
``DefaultAzureCredential`` chain is used (env vars, managed identity, Azure CLI
login, etc.).

Environment variables:
    AZURE_OPENAI_ENDPOINT      e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_VERSION   e.g. 2024-02-15-preview
    AZURE_OPENAI_CLIENT_ID     (optional) user-assigned identity / client id
    AZURE_OPENAI_DEPLOYMENT    chat/completions deployment name (read by callers)
"""

from __future__ import annotations

import os
from functools import lru_cache

from openai import AzureOpenAI

# Scope for Azure Cognitive Services (Azure OpenAI) data-plane access.
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
_DEFAULT_API_VERSION = "2024-02-15-preview"


# --------------------------------------------------------------------------- #
#  Fatal-error handling                                                        #
# --------------------------------------------------------------------------- #
# When something is wrong with the Azure OpenAI client itself — the deployment
# does not exist, we are being rate/token/quota limited, or auth fails — there
# is no point retrying the call or moving on to the next experiment. Every one
# of them will fail the same way. These helpers detect that situation so callers
# can abort the whole run immediately instead of grinding through it.


class FatalAzureError(RuntimeError):
    """Unrecoverable Azure OpenAI client error.

    Raised for problems that will affect *every* subsequent call — a missing or
    wrong deployment (``DeploymentNotFound``), rate limits (RPM), token limits
    (TPM), quota exhaustion, or authentication/authorization failures. Callers
    should let this propagate and stop the entire run rather than retrying or
    continuing to the next item.
    """


# Substrings (lower-cased) that identify a fatal Azure error even when it is
# wrapped in a generic Exception and the SDK type is not available.
_FATAL_ERROR_MARKERS: tuple[str, ...] = (
    "deploymentnotfound",
    "the api deployment for this resource does not exist",
    "deployment does not exist",
    "model_not_found",
    "rate limit",
    "ratelimit",
    "requests per minute",
    "tokens per minute",
    "token rate limit",
    "quota",
    "insufficient_quota",
    "exceeded call rate limit",
    "429",
    "invalid_api_key",
    "access denied",
    "permissiondenied",
    "unauthorized",
    "authenticationerror",
    "credential",
)


def is_fatal_azure_error(exc: BaseException) -> bool:
    """Return True if *exc* is an unrecoverable Azure OpenAI client error.

    "Fatal" means: the deployment is missing/misconfigured, we hit a rate/token/
    quota limit, or authentication failed — anything originating from the Azure
    client that will make every subsequent call fail the same way. Content-level
    problems (malformed/empty JSON in an otherwise successful response) are NOT
    fatal and are left for the caller's normal retry logic.
    """
    if isinstance(exc, FatalAzureError):
        return True

    # Azure identity / credential failures (keyless auth).
    try:
        from azure.core.exceptions import (
            ClientAuthenticationError,
            HttpResponseError,
        )

        if isinstance(exc, (ClientAuthenticationError, HttpResponseError)):
            return True
    except Exception:
        pass

    # Any error raised by the OpenAI/Azure OpenAI SDK is, by definition, related
    # to the Azure client — deployment, rate/token limits, auth, bad request,
    # connection/timeout, or server error. Treat all of them as fatal so a
    # broken client stops the run instead of silently retrying forever.
    try:
        import openai

        if isinstance(exc, openai.APIError):
            return True
    except Exception:
        pass

    # Fallback: inspect the message for known Azure error signatures (covers
    # errors that were re-wrapped in a generic Exception before reaching us).
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_ERROR_MARKERS)


def raise_if_fatal_azure_error(exc: BaseException) -> None:
    """Re-raise *exc* as a :class:`FatalAzureError` if it is unrecoverable.

    Does nothing for recoverable errors, so it is safe to call at the top of an
    ``except`` block before the normal retry/continue handling.
    """
    if is_fatal_azure_error(exc):
        if isinstance(exc, FatalAzureError):
            raise exc
        raise FatalAzureError(str(exc)) from exc


@lru_cache(maxsize=1)
def _token_provider():
    """Build a cached bearer-token provider for Azure OpenAI."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    client_id = os.getenv("AZURE_OPENAI_CLIENT_ID")
    if client_id:
        credential = DefaultAzureCredential(managed_identity_client_id=client_id)
    else:
        credential = DefaultAzureCredential()
    return get_bearer_token_provider(credential, _TOKEN_SCOPE)


def make_azure_openai_client() -> AzureOpenAI:
    """Return an AzureOpenAI client.

    Auth precedence:
      1. If ``AZURE_OPENAI_API_KEY`` is set, use key-based auth (used by the
         run environment, where the key is injected from a secret store as
         an environment variable).
      2. Otherwise, keyless (Microsoft Entra ID) auth via a bearer token.
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", _DEFAULT_API_VERSION)

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=api_key,
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=_token_provider(),
    )
