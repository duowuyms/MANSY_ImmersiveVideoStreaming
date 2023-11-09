import yaml
from munch import Munch


DEFAULT_CONFIG_YML_PATH = '../config.yml'


def get_config_from_yml(config_yml_path=None):
    if config_yml_path is None:
        config_yml_path = DEFAULT_CONFIG_YML_PATH
    
    with open(config_yml_path, 'r', encoding='utf8') as config_yml_file:
        config = yaml.load(config_yml_file, Loader=yaml.SafeLoader)
        config_yml_file.close()
        
    config = Munch(config)
    # concat datasets_base_dir with other datasets_dir
    datasets_dir_list = [config.raw_datasets_dir, config.raw_network_datasets_dir, config.viewport_datasets_dir,
                         config.video_datasets_dir, config.network_datasets_dir]
    for datasets_dir in datasets_dir_list:
        for key in datasets_dir.keys():
            datasets_dir[key] =  config.datasets_base_dir + datasets_dir[key]

    # concat results_base_dir with other results_dir
    config.vp_results_dir = config.results_base_dir + config.vp_results_dir
    config.bs_results_dir = config.results_base_dir + config.bs_results_dir

    # concat models_base_dir with other models_dir
    config.vp_models_dir = config.models_base_dir + config.vp_models_dir
    config.bs_models_dir = config.models_base_dir + config.bs_models_dir

    print(config)
    return config


if __name__ == '__main__':
    get_config_from_yml()
