# A simple example of using GPT-3

## Installation

First, find the secret OpenAI API key from [this](https://beta.openai.com/account/api-keys) page.

Then, create a text file named `API_KEYS` in this folder and put your OpenAI API key for access to GPT-3 models:

`export OPENAI_API_KEY=<your OpenAI API key>`

This file is in `.gitignore` since it is important that you do not accidentally commit your key.
Install the python packages specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the example

This example reads partial conversations from a text file (partial_dialogs.txt) and uses GPT-3 to generate continuations for them. It then classfies each new utterance based on being related/unrelated to the topic of "pets". This demonstrates bothe generation and classification capabilities of GPT-3.


Run the following command. The input variables are optional and if not provided, will assume their default values.

```
make self-chat engine=text-davinci-002 temperature=0.8
```

Please read through the Makefil and follow the Python code. Make sure you understand how the code works before using it for your project.
The `NeuralWorker` class is at the center of this example, and is meant to be reusable for text generation and classfication tasks.
IT calls `openai.Completion.create()` from the `openai` Python library.
In order to understand what this function does, read the [Create completion](https://beta.openai.com/docs/api-reference/completions/create) documentation.
