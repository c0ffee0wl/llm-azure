import os
from typing import Iterable, Iterator, List, Union, Optional

import click
import llm
import yaml
from llm import EmbeddingModel, hookimpl
from llm.default_plugins.openai_models import AsyncChat, Chat, _Shared, not_nulls
from openai import AsyncAzureOpenAI, AzureOpenAI
from llm.utils import remove_dict_none_values

DEFAULT_KEY_ALIAS = "azure"
DEFAULT_KEY_ENV_VAR = "AZURE_OPENAI_API_KEY"


def _ensure_config_file():
    filepath = llm.user_dir() / "azure" / "config.yaml"
    if not filepath.exists():
        filepath.parent.mkdir(exist_ok=True)
        filepath.write_text("[]")
    return filepath


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def azure():
        "Commands for working with azure models"

    @azure.command()
    def config_file():
        "Display the path to the azure config file"
        click.echo(_ensure_config_file())


@hookimpl
def register_models(register):
    azure_path = _ensure_config_file()
    with open(azure_path) as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models or []:
        if model.get('embedding_model'):
            continue

        needs_key = model.get("needs_key", DEFAULT_KEY_ALIAS)
        key_env_var = model.get("key_env_var", DEFAULT_KEY_ENV_VAR)
        aliases = model.pop("aliases", [])
        # Pass all relevant model parameters directly to AzureChat/AzureAsyncChat
        # This assumes the model dict contains 'model_id', 'model_name', 'api_base', 'api_version'
        # and optionally 'attachment_types', 'can_stream'.
        register(
            AzureChat(needs_key=needs_key, key_env_var=key_env_var, **model),
            AzureAsyncChat(needs_key=needs_key, key_env_var=key_env_var, **model),
            aliases=aliases,
        )


@hookimpl
def register_embedding_models(register):
    azure_path = _ensure_config_file()
    with open(azure_path) as f:
        azure_models = yaml.safe_load(f)

    for model in azure_models or []:
        if not model.get('embedding_model'):
            continue

        needs_key = model.get("needs_key", DEFAULT_KEY_ALIAS)
        key_env_var = model.get("key_env_var", DEFAULT_KEY_ENV_VAR)
        aliases = model.pop("aliases", [])
        model.pop('embedding_model') # Remove the flag before passing to constructor

        register(
            AzureEmbedding(needs_key=needs_key, key_env_var=key_env_var, **model),
            aliases=aliases,
        )


class AzureShared(_Shared):
    def __init__(self, model_id, model_name, api_base, api_version, attachment_types=None, can_stream=True, needs_key: str = DEFAULT_KEY_ALIAS, key_env_var: str = DEFAULT_KEY_ENV_VAR, **kwargs):
        # The base _Shared class expects model_id and possibly can_stream, and other kwargs
        super().__init__(model_id=model_id, model_name=model_name, can_stream=can_stream, needs_key=needs_key, key_env_var=key_env_var, **kwargs)
        self.api_base = api_base
        self.api_version = api_version
        self.attachment_types = attachment_types or set()

    def get_client(self, key, *, async_=False):
        kwargs = {
            "api_key": self.get_key(key),
            "api_version": self.api_version,
            "azure_endpoint": self.api_base,
        }
        if os.environ.get("LLM_OPENAI_SHOW_RESPONSES"):
            kwargs["http_client"] = self.logging_client()
        if async_:
            return AsyncAzureOpenAI(**kwargs)
        else:
            return AzureOpenAI(**kwargs)

    def build_kwargs(self, prompt, stream):
        kwargs = dict(not_nulls(prompt.options))
        json_object = kwargs.pop("json_object", None)
        if "max_tokens" not in kwargs and self.default_max_tokens is not None:
            kwargs["max_tokens"] = self.default_max_tokens
        if json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if stream:
            # For Azure OpenAI, stream_options is generally supported now
            # https://github.com/openai/openai-python/issues/1469 was resolved.
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    def execute(self, prompt, stream, response, conversation=None, async_override=False):
        messages = []
        current_system = None

        # Handle conversation history and attachments
        if conversation is not None:
            for prev_response in conversation.responses:
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    if (
                        prev_response.prompt.system
                        and prev_response.prompt.system != current_system
                    ):
                        messages.append(
                            {"role": "system", "content": prev_response.prompt.system},
                        )
                        current_system = prev_response.prompt.system
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt},
                    )

                messages.append(
                    {"role": "assistant", "content": prev_response.text_or_raise()}
                )

        # Handle system prompt
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})

        # Handle attachments for current prompt
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment))
            messages.append({"role": "user", "content": attachment_message})

        response._prompt_json = {"messages": messages}
        kwargs = self.build_kwargs(prompt, stream)
        client = self.get_client(key=None, async_=async_override) # key=None will make it use self.get_key() internally

        if async_override:
            async def async_generator():
                completion = await client.chat.completions.create(
                    model=self.model_name or self.model_id,
                    messages=messages,
                    stream=True,
                    **kwargs,
                )
                chunks = []
                async for chunk in completion:
                    chunks.append(chunk)
                    content = self.combine_chunks(chunks)
                    yield content
                response.response_json = remove_dict_none_values(
                    self.combine_chunks(chunks, usage=True)
                )
                response.usage = response.response_json.get("usage")
            return async_generator()
        else: # Synchronous execution
            completion = client.chat.completions.create(
                model=self.model_name or self.model_id,
                messages=messages,
                stream=stream, # Use the stream parameter for sync calls
                **kwargs,
            )
            if stream:
                chunks = []
                for chunk in completion:
                    chunks.append(chunk)
                    content = self.combine_chunks(chunks)
                    yield content
                response.response_json = remove_dict_none_values(
                    self.combine_chunks(chunks, usage=True)
                )
                response.usage = response.response_json.get("usage")
            else:
                response.response_json = remove_dict_none_values(completion.model_dump())
                yield completion.choices[0].message.content


class AzureChat(AzureShared, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self, prompt, stream, response, conversation=None):
        yield from super().execute(prompt, stream, response, conversation, async_override=False)


class AzureAsyncChat(AzureShared, AsyncChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def execute(self, prompt, stream, response, conversation=None):
        async for chunk in await super().execute(prompt, stream, response, conversation, async_override=True):
            yield chunk


class AzureEmbedding(EmbeddingModel):
    batch_size = 100

    def __init__(self, model_id, model_name, api_base, api_version, needs_key: str = DEFAULT_KEY_ALIAS, key_env_var: str = DEFAULT_KEY_ENV_VAR, **kwargs):
        super().__init__(model_id=model_id, model_name=model_name, needs_key=needs_key, key_env_var=key_env_var, **kwargs)
        self.api_base = api_base
        self.api_version = api_version

    def embed_batch(self, items: Iterable[Union[str, bytes]]) -> Iterator[List[float]]:
        client = AzureOpenAI(
            api_key=self.get_key(),
            api_version=self.api_version,
            azure_endpoint=self.api_base,
        )
        kwargs = {
            "input": items,
            "model": self.model_name,
        }
        results = client.embeddings.create(**kwargs).data
        return ([float(r) for r in result.embedding] for result in results)


def _attachment(attachment):
    url = attachment.url
    base64_content = ""
    if not url or attachment.resolve_type().startswith("audio/"):
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    if attachment.resolve_type().startswith("image/"):
        return {"type": "image_url", "image_url": {"url": url}}
    else:
        format_ = "wav" if attachment.resolve_type() == "audio/wav" else "mp3"
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64_content,
                "format": format_,
            },
        }
