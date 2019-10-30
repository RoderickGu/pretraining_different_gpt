import argparse
import regex as re

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    # add all arguments
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                            help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    # logging
    parser.add_argument('--logging_steps', type=int, default=50,
                            help="Log every X updates steps.")

    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")

    # process all arguments
    args = parser.parse_args()
    return args


def safe_clean_text(text):
    """ This is a safe text cleaning procedure for all datasets
    """
    # weird words
    text = text.replace("¡ª", "")
    
    # handle \\\'t \\\'ve
    text = text.replace("\\", "")
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # fix puncutations
    text = text.replace(" ?", "?")
    text = text.replace(" .", ".")
    text = text.replace(" ,", ",")
    text = text.replace(" !", "!")
    
    return text


class DialogFragmentSampler:
    def __init__(self, max_len=1024):
        """Sample dialog fragments from a dialog
        """
        self.max_tokens_len = max_len
        
    def __call__(self, dialog):
        """dialog is a dict which has key "token_ids"
        """
        dialog_fragment = {}
        
        lengths = np.array([len(item) for item in dialog['token_ids']])
            
        # if the entire dialog is smaller than the max len
        if lengths.sum() < self.max_tokens_len:
            return dialog
            
        cumsum_len =  lengths.cumsum()
        reverse_cumsum_len = cumsum_len[-1] - cumsum_len
        
        # based on the reverse cumsum, we can have a range to select from
        start_turns = np.arange(len(reverse_cumsum_len))[reverse_cumsum_len > self.max_tokens_len]
        # remove odd numbers
        start_turns = [idx for idx in start_turns if idx%2==0]
        # randomly choose one
        random_start_turn = random.choice(start_turns)
        new_cumsum_len = cumsum_len - cumsum_len[random_start_turn]
        
        # find the maximum end turn (only odd turn)
        for i in reversed(range(len(new_cumsum_len))):
            if i%2==1 and new_cumsum_len[i] < self.max_tokens_len:
                random_end_turn = i
                break
        
        dialog_fragment["text"] = dialog['text'][random_start_turn:random_end_turn+1]
        dialog_fragment["token_ids"] = dialog['token_ids'][random_start_turn:random_end_turn+1]
        
        return dialog_fragment