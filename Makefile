include ./API_KEYS

### Input variables and their default values
engine ?= text-curie-001
num_inputs ?= 2
num_input_turns ?= 5
num_output_turns ?= 10
temperature ?= 0.7

.PHONY: self-chat

self-chat: self_chat.py prompts/chat.prompt prompts/is_pets.prompt partial_dialogs.txt
	python self_chat.py \
		--generation_prompt_template_file prompts/chat.prompt \
		--classification_prompt_template_file prompts/is_pets.prompt \
		--input_file partial_dialogs.txt \
		--engine $(engine) \
		--output_file output.txt \
		--temperature $(temperature) \
    	--top_p 0.9 \
    	--frequency_penalty 0.0 \
    	--presence_penalty 0.1 \
		--num_inputs $(num_inputs) \
		--num_input_turns $(num_input_turns) \
		--num_output_turns $(num_output_turns)