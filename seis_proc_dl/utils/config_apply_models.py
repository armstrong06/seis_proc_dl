import json

"""Copied this class from https://github.com/The-AI-Summer/Deep-Learning-In-Production"""
# paths, models, output_file_names, swag_info, options
class Config():
    """Config class which contains data, train, and model hyperparameters"""
    def __init__(self, model_params, paths, models, output_file_names, swag_info, options):
        self.model_params = model_params
        self.paths = paths
        self.models = models
        self.output_file_names = output_file_names
        self.swag_info = swag_info
        self.options = options

    @classmethod
    def from_json(cls, cfg):
        """Creates config from json"""
        params = json.loads(json.dumps(cfg), object_hook=HelperObject)
        return cls(params.model_params, params.paths, params.models, 
                    params.output_file_names, params.swag_info, params.options)

class HelperObject(object):
    """Helper class to convert json into Python object"""
    def __init__(self, dict_):
        self.__dict__.update(dict_)