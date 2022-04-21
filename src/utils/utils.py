import ast
from collections import ChainMap
import yaml

def evalfn(pairs):
    """
    Used for parsing command line arguments
    """
    res = {}
    for key, val in pairs:
        if val in {'true','false'}:
            res[key] = val == 'true'
            continue
        try:
            res[key] = ast.literal_eval(val)
        except Exception as e:
            res[key] = val
    return res

################################################################
# configs
################################################################


def get_config_cmd_yaml_default(cmd_args, config_defaults_dict):
    """
    Return a config dict with parameters from (in order of overriding):
    * Command-line arguments: python3 script.py 9 arg1=3 arg2=4.5 arg3=True arg4="fourier"
    * YAML config file named "config_009.yml" (using the number from the first command-line argument)
    * Dictionary of default parameters defined in this script

    Parameters
    ----------
    cmd_args :
        Arguments from command line, should be equal to sys.argv
    config_defaults_dict :
        Default config parameters defined in this script

    Returns
    -------
    config_dictionary
        Dictionary of the merged config parameters
    """

    # CMD parameters dictionary
    # cmd_args_dict = dict(arg.split('=') for arg in cmd_args[2:])
    # for key, value in cmd_args_dict.items():
    #     print(key, ' : ', value)
    # typed_cmd_args_dict = evalfn(list(cmd_args_dict.items()))
    # for key, value in typed_args_dict.items():
    #     print(key, ' : ', value, " - ", type(value))
    # print("CMD parameters: ")
    # print(typed_cmd_args_dict)

    # # CMD parameters dictionary + YAML Config file dictionary
    config_file_number = 0  # default config file number
    if len(cmd_args) > 1:
        if (
            "=" not in cmd_args[1]
        ):  # if first arg is not a parameter, then it is the config file number
            config_file_number = cmd_args[1]
            cmd_args_dict = dict(arg.split("=") for arg in cmd_args[2:])
        else:
            cmd_args_dict = dict(arg.split("=") for arg in cmd_args[1:])

    typed_cmd_args_dict = evalfn(list(cmd_args_dict.items()))
    print("CMD parameters: ")
    print(typed_cmd_args_dict)

    # print(config_file_number, " ", type(config_file_number))
    padded_config_file_number = str(config_file_number).zfill(3)
    CONFIG_FILE_PATH = f"config_{padded_config_file_number}.yml"

    with open(CONFIG_FILE_PATH) as f:
        config_file_dict = yaml.load(f, Loader=yaml.FullLoader)
        print("Config file parameters: ")
        print(config_file_dict)

    # Default dict
    print("Default script parameters: ")
    print(config_defaults_dict)

    # Gathering into 1 unique dictionary
    # Priority order: 1/ CMD, 2/ Config file, 3/ Default_dict
    config_dictionary = dict(
        ChainMap(typed_cmd_args_dict, config_file_dict, config_defaults_dict)
    )

    print("CMD + Config file + Default: ")
    print(config_dictionary)

    return config_dictionary

