import yaml


def save_cfg_to_yaml(cfg_dict, cfg_path):
    """ Save a config to a YAML file. """
    with open(cfg_path, 'w+') as fw:
        yaml.dump(cfg_dict, fw)


def load_cfg_from_yaml(cfg_path):
    """ Load a config from a YAML file. """
    with open(cfg_path, 'r+') as fr:
        cfg_as_dict = yaml.load(fr, Loader=yaml.FullLoader)
        print("Load config from {}".format(cfg_path))
        print("Loaded config: {}".format(cfg_as_dict))
    return cfg_as_dict


def merge_cfg_from_args(cfg_as_dict, args):
    """ Merge a config from python argparser. """
    for item in args._get_kwargs():
        cfg_as_dict[item[0]] = item[1]
    return cfg_as_dict


# def main():
#     # config_dict = {
#     #     'MODEL':
#     #         {'NAME': 'ResNet50'},
#     #     'DATASET':
#     #         {'LABEL_SIZE': 1, 'IMAGE_SIZE': 32, 'NUM_CHANNELS': 3, 'PIXEL_DEPTH': 255, 'NUM_CLASSES': 10,
#     #          'TRAIN_NUM': 10000, 'TRAIN_NUMS': 50000, 'TEST_NUM': 10000}
#     # }
#     config_dict = load_cfg_from_yaml('config/CIFAR10/R_50.yaml')
#     save_cfg_to_yaml(config_dict, 'config/CIFAR10/R_50_test.yaml')
#     config_dict_load = load_cfg_from_yaml('config/CIFAR10/R_50_test.yaml')
#
#
# if __name__ == '__main__':
#     main()
