import json
import time
def log_training(data_log):
    return {
        "data": data_log, 
    }
def print_name(list):
    return [{"func": item["func"].__name__, "params" : item["params"]} for item in list]

def json_to_file(file_name, data):
    with open(file_name + time.strftime("%Y%m%d-%H%M%S") + ".json", 'w') as f:
        json.dump(data.__dict__, f, indent=2)


class Training_Data_Log:
    model_config = None
    model_config_method_string = None
    training_sets = None
    training_set_size = None
    validation_set_size = None
    test_set_size = None
    image_pre_processors = None
    noise_added = None
    training_data_masks = None
    model_summary = None
    model_optimizer = None
    model_history = None
    model_history_final_epoch = None
    results = None

    def _get_total_epochs(self):
        return self.__total_epochs
    def _set_total_epochs(self, value):
        if not isinstance(value, int):
            raise TypeError("total_epochs must be set to an integer")
        self.__total_epochs = value

    total_epochs = property(_get_total_epochs, _set_total_epochs)