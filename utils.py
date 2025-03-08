DATA_MAP = [
    {
        "path":"/home/mltrain/data/compound_words.pyth.24.-1" # 0
    },
    {
        "path":"/home/mltrain/data/distribution_id.pyth.512.-1" # 1
    },
    {
        "path":"/home/mltrain/data/ewt.pyth.512.-1" # 2
    },
    {
        "path":"/home/mltrain/data/latex.pyth.1024.-1" # 3
    },
    {
        "path":"/home/mltrain/data/natural_lang_id.pyth.512.-1" # 4
    },
    {
        "path":"/home/mltrain/data/programming_lang_id.pyth.512.100" # 5
    },
    {
        "path":"/home/mltrain/data/text_features.pyth.256.10000" # 6
    },
    {
        "path":"/home/mltrain/data/wikidata_sorted_is_alive.pyth.128.6000" # 7
    },
    {
        "path":"/home/mltrain/data/wikidata_sorted_occupation.pyth.128.6000" # 8
    },
    {
        "path":"/home/mltrain/data/wikidata_sorted_occupation_athlete.pyth.128.5000" # 9
    },
    {
        "path":"/home/mltrain/data/wikidata_sorted_political_party.pyth.128.3000" # 10
    },
    {
        "path":"/home/mltrain/data/wikidata_sorted_sex_or_gender.pyth.128.6000", # 11
    }
    
]


def get_data_path(dataset_idx):
    return DATA_MAP[dataset_idx]['path']


def preprocess_data(sample, dataset_idx):
    # sex_gender | political party | athlete | alive
    if dataset_idx == 11 or dataset_idx == 10 or dataset_idx == 9 or dataset_idx == 7 or dataset_idx == 8:
        # probe only the name of the person: from name_index_start to name_index_finish (inclusive)
        tokens = sample['tokens']
        name_range = list(range(sample['name_index_start'],sample['name_index_end']+1))
        return tokens, name_range #name range are the token positions to probe
    elif dataset_idx == 4:
        tokens = sample['tokens']
        pos = sample['valid_indices']
        return tokens, pos
    else:
        raise NotImplementedError(f"Preprocessing for {dataset_idx} is not implemented")
        
        
        
    