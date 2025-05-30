# llm-azure

[![PyPI](https://img.shields.io/pypi/v/llm-azure.svg)](https://pypi.org/project/llm-azure/)
[![Changelog](https://img.shields.io/github/v/release/fabge/llm-azure?include_prereleases&label=changelog)](https://github.com/fabge/llm-azure/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/fabge/llm-azure/blob/main/LICENSE)

LLM access to the Azure OpenAI SDK

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

```bash
llm install llm-azure
```

## Usage

First, set an API key for Azure OpenAI. By default, the plugin looks for a key aliased as `azure` (e.g., from `llm keys set azure`) or in the `AZURE_OPENAI_API_KEY` environment variable.

```bash
llm keys set azure
# Paste key here
```

To add the `gpt-4-32k` chat model, and embedding model `text-embedding-3-small` deployed in your Azure Subscription, add this to your `azure/config.yaml` file:

```yaml
- model_id: gpt-4-32k
  model_name: gpt-4-32k
  api_base: https://deployment.openai.azure.com/
  api_version: '2023-05-15'

- model_id: text-embedding-3-small
  embedding_model: true
  model_name: text-embedding-3-small
  api_base: https://deployment.openai.azure.com/
  api_version: '2023-05-14'
```

The configuration file should be in the `azure` directory in the config of your `llm` installation.
Run this command to find the location of the config file:

```bash
llm azure config-file
```

or you can open the file with:

```bash
open "$(llm azure config-file)"
```

The `model_id` is the name LLM will use for the model. The `model_name` is the name which needs to be passed to the API - this might differ from the `model_id`, especially if `model_id` could potentially clash with other installed models.

### Attachments

To enable the `-a` flag for models that support multi-modal input (like image or audio attachments), add an `attachment_types` section to your `config.yaml` with the desired MIME types (subject to the underlying model's support). Expanding on the example above, your `config.yaml` would now look like this:

```yaml
- model_id: gpt-4-32k
  model_name: gpt-4-32k # Corrected from deployment_name
  api_base: https://deployment.openai.azure.com/
  api_version: '2023-05-15'
  attachment_types:
    - "image/png"
    - "image/jpeg"
    - "audio/wav"
    - "audio/mp3"
```

### Streaming

For models that can stream responses, add `can_stream: true` to their configuration:

```yaml
- model_id: gpt-4-32k
  model_name: gpt-4-32k
  api_base: https://deployment.openai.azure.com/
  api_version: '2023-05-15'
  can_stream: true
```

### Customizing API Key Configuration

By default, `llm-azure` models will look for an API key aliased as `azure` (from `llm keys set azure`) or in the `AZURE_OPENAI_API_KEY` environment variable. You can override these defaults for individual models by adding `needs_key` and `key_env_var` to your `config.yaml` entry. This is useful if you manage multiple Azure subscriptions or different sets of API keys.

For example, to configure a model to use a key aliased as `my-other-azure-key` (set via `llm keys set my-other-azure-key`) or from an environment variable named `MY_AZURE_API_KEY`:

```yaml
- model_id: my-special-gpt
  model_name: my-deployment-name
  api_base: https://your_other_[deployment.openai.azure.com/](https://deployment.openai.azure.com/)
  api_version: '2023-05-15'
  needs_key: my-other-azure-key
  key_env_var: MY_AZURE_API_KEY

- model_id: another-embedding
  embedding_model: true
  model_name: other-embedding-deployment
  api_base: https://your_other_[deployment.openai.azure.com/](https://deployment.openai.azure.com/)
  api_version: '2023-05-14'
  needs_key: my-other-azure-key
  key_env_var: MY_AZURE_API_KEY
```
