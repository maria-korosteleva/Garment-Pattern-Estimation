"""In simulated dataset, filter only samples which parameter passes the filter """

import os

import customconfig
from pattern.wrappers import VisPattern
import json


def isAllowed(pattern, param_filter):
    for param in param_filter:
        value = pattern.parameters[param]['value']
        if value < param_filter[param][0] or value > param_filter[param][1]:
            return False
    return True


datasets = [
    'jacket_hood_2700',
    'dress_sleeveless_2550',
    'jacket_2200',
    'jumpsuit_sleeveless_2000',
    'pants_straight_sides_1000',
    'skirt_2_panels_1200',
    'skirt_4_panels_1600',
    'skirt_8_panels_1000',
    'tee_2300',
    'tee_sleeveless_1800',
    'wb_dress_sleeveless_2600',
    'wb_pants_straight_1500',
    'test/dress_150',
    'test/jacket_hood_sleeveless_150',
    'test/jacket_sleeveless_150',
    'test/jumpsuit_150',
    'test/skirt_waistband_150',
    'test/tee_hood_150',
    'test/wb_jumpsuit_sleeveless_150'
]

system_props = customconfig.Properties('./system.json')
filter_file = './nn/data_configs/param_filter.json'
with open(filter_file, 'r') as f:
    filters = json.load(f)

counts = {}
filtered_counts = {}
for dataset in datasets:
    datapath = os.path.join(system_props['datasets_path'], dataset)
    dataset_props = customconfig.Properties(os.path.join(datapath, 'dataset_properties.json'))
    template_name = dataset_props['templates'].split('/')[-1].split('.')[0]

    counts[dataset] = 0
    filtered_counts[dataset] = 0
    for root, dirs, files in os.walk(datapath):
        for filename in files:
            if 'specification' in filename:
                pattern = VisPattern(os.path.join(root, filename))
                # check parameters!
                if isAllowed(pattern, filters[template_name]):
                    filtered_counts[dataset] += 1

                counts[dataset] += 1

    print(f'{dataset}::{filtered_counts[dataset]} of {counts[dataset]}')
