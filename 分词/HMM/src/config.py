config_instance = None


class Config:

    def __init__(self):
        self.config_dict = {
            'segment': {
                'train_corpus_path': 'data/pcs_corpus.txt',

                'hmm_model_path': 'data/hmm_pcs.pkl',
            },
        }

    def get(self, section_name, arg_name):
        return self.config_dict[section_name][arg_name]


def get_config():
    global config_instance
    if not config_instance:
        config_instance = Config()
    return config_instance
