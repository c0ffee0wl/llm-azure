# llm-azure

[![PyPI](https://img.shields.io/pypi/v/llm-azure.svg)](https://pypi.org/project/llm-azure/)
[![Changelog](https://img.shields.io/github/v/release/fabge/llm-azure?include_prereleases&label=changelog)](https://github.com/fabge/llm-azure/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/fabge/llm-azure/blob/main/LICENSE)

LLM access to the Azure OpenAI SDK

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).

Because this plugin is a community fork, it is not published under the upstream project's name.  Therefore to run this fork, you must ]manually install the plugin as a plugin developer would](https://llm.datasette.io/en/stable/plugins/tutorial-model-plugin.html#installing-your-plugin-to-try-it-out):


```bash
git clone https://github.com/bexelbie/llm-azure.git
cd llm-azure
llm install -e .
# llm install llm-azure
```

## Usage

First, set an API key for Azure OpenAI. By default, the plugin looks for a key aliased as `azure` (e.g., from `llm keys set azure`) or in the `AZURE_OPENAI_API_KEY` environment variable.

```bash
llm keys set azure
# Paste key here
```

### Configuration

To register Azure OpenAI models with LLM, you need to create a configuration file. This file should be named `config.yaml` and placed in the `azure` directory within your LLM configuration folder.

You can find the location of this directory and open the file using these commands:

```bash
llm azure config-file
# or to open directly:
open "$(llm azure config-file)"
```

The `config.yaml` file should contain a list of model configurations. Each entry defines a model and its properties.

**For comprehensive examples and detailed explanations of all available configuration options, please refer to the `example-config.yaml` file in this repository.** This file demonstrates how to configure:

*   **Chat Models:** Basic setup, streaming, multi-modal input (attachments), vision, audio, reasoning, and system prompt capabilities.
*   **Embedding Models:** How to register models for embeddings.
*   **Custom API Keys:** Using different API key aliases or environment variables for specific models.

The `model_id` is the name LLM will use for the model (e.g., `llm -m <model_id>`). The `model_name` is the name of your deployment in Azure OpenAI Studio, which is passed to the API.

# Fork Notes

This fork brings `llm-azure` up-to-date with community proposed patches and forks. Specifically, this fork has currently folded in changes proposed or made by:
https://github.com/kj9/llm-azure - config file management and cli improvements
https://github.com/kmad0/llm-azure - Add attachment support for Azure chat models
https://github.com/While the patch in laszlovandenhoek/llm-azure wasn't needed, it did inspire an update to the README for can_stream
https://github.com/0gust1/llm-azure - allow for selective overriding of Azure API Keys for differing deployments
https://github.com/jonasherfot/llm-azure - support vision, audio, reasoning, and system prompt capabilities
