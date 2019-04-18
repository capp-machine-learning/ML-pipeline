'''
Config for Machine Learning Analysis.

Si Young Byun (syb234)
'''
PIPELINE_CONFIG = {
    'dataset': 'data/credit-data.csv',
    'data_dict': 'data/data-dictionary.xls',
    'outcome_var': 'SeriousDlqin2yrs',
    'test_size': 0.3,
    'threshold': 0.4,
    'random_state': 10
}