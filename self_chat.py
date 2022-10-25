"""
GPT-3 converse with itself continuing from partial datasets
"""

from typing import List
import argparse
from neural_worker import NeuralWorker
from tqdm import tqdm


def dialog_history_to_text(history: List[str], they='They', you='You') -> str:
    """
    The last turn always starts with "You: "
    """
    ret = ''
    if len(history) % 2 == 1:
        ret += they+': ' + history[0]
        history = history[1:]
    for i in range(len(history) // 2):
        ret += '\n'+you+': ' + history[2*i]
        ret += '\n'+they+': ' + history[2*i+1]

    # remove the extra starting newline
    if ret[0] == '\n':
        ret = ret[1:]

    return ret

def write_dialog_history_to_file(history, output_file):
    output_file.write('=====\n')
    for turn_id, turn in enumerate(history):
        output_file.write('Person ' + str((turn_id % 2)+1) + ': ' + turn + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation_prompt_template_file', type=str, required=True,
                        help='The path to the file containing the GPT-3 prompt.')
    parser.add_argument('--classification_prompt_template_file', type=str, required=True,
                        help='The path to the file containing the GPT-3 prompt.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Where to read the partial conversations from.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Where to write the outputs.')
    parser.add_argument('--engine', type=str, required=True,
                        choices=['ada',
                                 'text-ada-001',
                                 'babbage',
                                 'text-babbage-001',
                                 'curie',
                                 'text-curie-001',
                                 'davinci',
                                 'text-davinci-001',
                                 'text-davinci-002'],
                        help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model

    parser.add_argument('--num_inputs', type=int, default=1, required=False, help='Number of dialogs to read from the input file (default: 1')
    parser.add_argument('--num_input_turns', type=int, default=2, required=False, help='Maximum number of turns to to use from each input dialog (default: 2')
    parser.add_argument('--num_output_turns', type=int, default=1, required=False, help='Number of turns to continue each dialog for (default: 1')

    # GPT-3 generation hyperparameters
    parser.add_argument('--max_tokens', type=int, default=40, required=False, help='')
    parser.add_argument('--temperature', type=float, default=0.8, required=False, help='')
    parser.add_argument('--top_p', type=float, default=0.9, required=False, help='')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, required=False, help='')
    parser.add_argument('--presence_penalty', type=float, default=0.0, required=False, help='')
    parser.add_argument('--stop_tokens', nargs='+', type=str,
                        default=None, required=False, help='Stop tokens for generation')

    args = parser.parse_args()

    all_dialogs = []
    with open(args.input_file) as input_file:
        dialog_history = []
        for line in input_file:
            if line.startswith('====='):
                all_dialogs.append(dialog_history)
                dialog_history = []
                if len(all_dialogs) < args.num_inputs:
                    continue
                else:
                    break
            dialog_history.append(line.strip())

    # initialize the NeuralWorkers
    generator_neural_worker = NeuralWorker(prompt_template_file=args.generation_prompt_template_file, engine=args.engine)
    classifier_neural_worker = NeuralWorker(prompt_template_file=args.classification_prompt_template_file, engine=args.engine)

    with open(args.output_file, 'w') as output_file:
        for dlg in tqdm(all_dialogs):
            new_dlg = dlg[:args.num_input_turns]
            for _ in range(args.num_output_turns):
                filled_prompt = generator_neural_worker.fill_prompt_template(history=dialog_history_to_text(new_dlg))
                reply = generator_neural_worker.generate(input_text=filled_prompt, args=args, postprocess=True, max_tries=1)
                if len(reply) == 0:
                    # handle the case where the output of GPT-3 only contains whitespace, so the above function returns and empty string
                    print('Empty generated output. Stopping the dialog early.')
                    break
                new_dlg.append(reply)

                # Find out if the utterance is about pets or not
                filled_prompt = classifier_neural_worker.fill_prompt_template(utterance=reply)
                is_pet_probability = classifier_neural_worker.classify(input_text=filled_prompt)
                print('The probability of this utterance being about pets is %.3f' % is_pet_probability)

            write_dialog_history_to_file(new_dlg, output_file)