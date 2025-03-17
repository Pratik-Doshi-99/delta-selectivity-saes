import torch
import os
import glob

PYTHIA_410M_ACTIVATIONS_ROOT = '/home/mltrain/delta-selectivity-saes/activations_pythia_410m'
PYTHIA_160M_ACTIVATIONS_ROOT = '/home/mltrain/delta-selectivity-saes/activations_pythia_160m'

DATA_MAP = [
    {
        "activation_sub_directory":"compound_word",
        'name':"compound_words",
        "path":"/home/mltrain/delta-selectivity-saes/data/compound_words.pyth.24.-1" # 0
        
    },
    {
        "activation_sub_directory":"data_subset",
        'name':"data_subset",
        "path":"/home/mltrain/delta-selectivity-saes/data/distribution_id.pyth.512.-1" # 1
    },
    {
        "activation_sub_directory":"ewt",
        'name':"ewt",
        "path":"/home/mltrain/delta-selectivity-saes/data/ewt.pyth.512.-1" # 2
    },
    {
        "activation_sub_directory":"latex",
        'name':"latex",
        "path":"/home/mltrain/delta-selectivity-saes/data/latex.pyth.1024.-1" # 3
    },
    {
        "activation_sub_directory":"nat_lang",
        'name':"nat_lang",
        "path":"/home/mltrain/delta-selectivity-saes/data/natural_lang_id.pyth.512.-1" # 4
    },
    {
        "activation_sub_directory":"programming_lang",
        'name':"programming_lang",
        "path":"/home/mltrain/delta-selectivity-saes/data/programming_lang_id.pyth.512.100" # 5
    },
    {
        "activation_sub_directory":"text_features",
        'name':"text_features",
        "path":"/home/mltrain/delta-selectivity-saes/data/text_features.pyth.256.10000" # 6
    },
    {
        "activation_sub_directory":"wikidata_is_alive",
        'name':"wikidata_is_alive",
        "path":"/home/mltrain/delta-selectivity-saes/data/wikidata_sorted_is_alive.pyth.128.6000" # 7
    },
    {
        "activation_sub_directory":"wikidata_occupation",
        'name':"wikidata_occupation",
        "path":"/home/mltrain/delta-selectivity-saes/data/wikidata_sorted_occupation.pyth.128.6000" # 8
    },
    {
        "activation_sub_directory":"wikidata_athlete",
        'name':"wikidata_athlete",
        "path":"/home/mltrain/delta-selectivity-saes/data/wikidata_sorted_occupation_athlete.pyth.128.5000" # 9
    },
    {
        "activation_sub_directory":"wikidata_politics",
        'name':"wikidata_politics",
        "path":"/home/mltrain/delta-selectivity-saes/data/wikidata_sorted_political_party.pyth.128.3000" # 10
    },
    {
        "activation_sub_directory":"wikidata_gender",
        'name':"wikidata_gender",
        "path":"/home/mltrain/delta-selectivity-saes/data/wikidata_sorted_sex_or_gender.pyth.128.6000", # 11
    }
    
]


def get_data_path(dataset_idx):
    return DATA_MAP[dataset_idx]['path']

def get_dataset_name(dataset_idx):
    return DATA_MAP[dataset_idx]['name']

def get_datasets():
    return [0,1,2,3,4,6,7,8,9,10,11]

def get_models():
    return ['EleutherAI_pythia-160m','EleutherAI_pythia-410m']

def get_layers(model):
    # have not configured all layers
    if model == 'EleutherAI_pythia-410m':
        return [0,1,2,10,11,12,21,22,23]
    elif model == 'EleutherAI_pythia-160m':
        return [0,1,5,6,10,11]
    else:
        raise NotImplementedError(f"Layers for {model} not configured")



def get_tensors(dir_pattern):
    all_files = glob.glob(dir_pattern)
    all_activations = []
    for file in all_files:
        checkpoint_model = torch.load(file, map_location=torch.device(device), weights_only=False)
        for i in range(len(checkpoint_model)):
            if checkpoint_model[i].is_sparse:
                checkpoint_model[i] = checkpoint_model[i].to_dense()
        all_activations += checkpoint_model
    return all_activations
    


def get_model_activation(model, layer, dataset_idx, device):
    dir_pattern = None
    if model == 'EleutherAI_pythia-160m':
        dir_pattern = os.path.join(PYTHIA_160M_ACTIVATIONS_ROOT, DATA_MAP[dataset_idx],"model-activations",f"*_{layer}L_*.pt")
    elif model == 'EleutherAI_pythia-410m':
        dir_pattern = os.path.join(PYTHIA_410M_ACTIVATIONS_ROOT, DATA_MAP[dataset_idx],"model-activations",f"*_{layer}L_*.pt")
    else:
        raise NotImplementedError(f"{model} not supported")
    
    return get_tensors(dir_pattern)
    
        

def get_sae_activations(model, layer, dataset_idx, device):
    dir_pattern = None
    if model == 'EleutherAI_pythia-160m':
        dir_pattern = os.path.join(PYTHIA_160M_ACTIVATIONS_ROOT, DATA_MAP[dataset_idx],"sae-activations",f"*_{layer}L_*.pt")
    elif model == 'EleutherAI_pythia-410m':
        dir_pattern = os.path.join(PYTHIA_410M_ACTIVATIONS_ROOT, DATA_MAP[dataset_idx],"sae-activations",f"*_{layer}L_*.pt")
    else:
        raise NotImplementedError(f"{model} not supported")
    
    return get_tensors(dir_pattern)
    

def get_binary_probes(sample, dataset_idx):
    # return a list: [(probe_name, probe_pos, target_value)]
    # if probe_pos is None, all captured pos (from preprocess_data) must be accounted
    if dataset_idx == 11:
        return [('wikidata_gender_1female_0male', None, 1 if sample['class'] == 'female' else 0)]
    elif dataset_idx == 10:
        return [('wikidata_politics_1democratic_0republic', None, 1 if sample['class'] == 'Democratic Party' else 0)]
    elif dataset_idx == 9:
        return [('wikidata_athlete_is_football', None, 1 if sample['class'] == 'association football player' else 0),
                ('wikidata_athlete_is_basketball', None, 1 if sample['class'] == 'basketball player' else 0),
                ('wikidata_athlete_is_baseball', None, 1 if sample['class'] == 'baseball player' else 0),
                ('wikidata_athlete_is_american_football', None, 1 if sample['class'] == 'American football player' else 0),
                ('wikidata_athlete_is_icehockey', None, 1 if sample['class'] == 'ice hockey player' else 0)]
    elif dataset_idx == 7:
        return [('wikidata_1alive_0dead', None, 1 if sample['class'] is True else 0)]
    elif dataset_idx == 8:
        return [('wikidata_occupation_is_singer', None, 1 if sample['class'] == 'singer' else 0),
                ('wikidata_athlete_is_actor', None, 1 if sample['class'] == 'actor' else 0),
                ('wikidata_athlete_is_politician', None, 1 if sample['class'] == 'politician' else 0),
                ('wikidata_athlete_is_journalist', None, 1 if sample['class'] == 'journalist' else 0),
                ('wikidata_athlete_is_athlete', None, 1 if sample['class'] == 'athlete' else 0),
                ('wikidata_athlete_is_researcher', None, 1 if sample['class'] == 'researcher' else 0)]
    elif dataset_idx == 4:
        return [('nat_lang_is_spanish', None, 1 if sample['class_ids'] == 1 else 0),
                ('nat_lang_is_english', None, 1 if sample['class_ids'] == 2 else 0),
                ('nat_lang_is_french', None, 1 if sample['class_ids'] == 3 else 0),
                ('nat_lang_is_dutch', None, 1 if sample['class_ids'] == 4 else 0),
                ('nat_lang_is_italian', None, 1 if sample['class_ids'] == 5 else 0),
                ('nat_lang_is_greek', None, 1 if sample['class_ids'] == 6 else 0),
                ('nat_lang_is_german', None, 1 if sample['class_ids'] == 7 else 0),
                ('nat_lang_is_portuguese', None, 1 if sample['class_ids'] == 8 else 0),
                ('nat_lang_is_swedish', None, 1 if sample['class_ids'] == 9 else 0)]
    
    elif dataset_idx == 1:
        return [('data_subset_is_wikipedia', None, 1 if sample['distribution'] == 'wikipedia' else 0),
                ('data_subset_is_pubmed_abstracts', None, 1 if sample['distribution'] == 'pubmed_abstracts' else 0),
                ('data_subset_is_stack_exchange', None, 1 if sample['distribution'] == 'stack_exchange' else 0),
                ('data_subset_is_github', None, 1 if sample['distribution'] == 'github' else 0),
                ('data_subset_is_arxiv', None, 1 if sample['distribution'] == 'arxiv' else 0),
                ('data_subset_is_uspto', None, 1 if sample['distribution'] == 'uspto' else 0),
                ('data_subset_is_freelaw', None, 1 if sample['distribution'] == 'freelaw' else 0),
                ('data_subset_is_hackernews', None, 1 if sample['distribution'] == 'hackernews' else 0),
                ('data_subset_is_enron', None, 1 if sample['distribution'] == 'enron' else 0)]
    
    else:
        raise NotImplementedError(f"Binary probes for {dataset_idx} is not implemented")



def preprocess_data(sample, dataset_idx):
    # returns all_tokens and captured tokens
    # sex_gender | political party | athlete | alive
    if dataset_idx == 11 or dataset_idx == 10 or dataset_idx == 9 or dataset_idx == 7 or dataset_idx == 8:
        # probe only the name of the person: from name_index_start to name_index_finish (inclusive)
        tokens = sample['tokens']
        name_range = list(range(sample['name_index_start'],sample['name_index_end']+1))
        return tokens, name_range #name range are the token positions to probe
    elif dataset_idx == 4:
        tokens = sample['tokens']
        # probe_indices + ~50 other random indices to probe
        pos = list(set(torch.randint(low=0, high=len(sample['tokens']), size=(55,)).tolist() + sample['probe_indices'].tolist()))
        return tokens, pos
    elif dataset_idx == 6:
        tokens = sample['tokens']
        pos = list(
                set(
                    torch.concat([
                        sample['contains_digit|probe_indices'],sample['all_digits|probe_indices'], sample['contains_capital|probe_indices'],
                        sample['leading_capital|probe_indices'],sample['all_capitals|probe_indices'],sample['contains_whitespace|probe_indices'],
                        sample['has_leading_space|probe_indices'],sample['no_leading_space_and_loweralpha|probe_indices'],sample['contains_all_whitespace|probe_indices'],
                        sample['is_not_alphanumeric|probe_indices'],sample['is_not_ascii|probe_indices']]
                    ).tolist()
                )
        )
        return tokens, pos
    elif dataset_idx == 1:
        tokens = sample['tokens']
        # probe_indices + ~50 other random indices to probe
        pos = list(set(torch.randint(low=0, high=len(sample['tokens']), size=(55,)).tolist() + sample['probe_indices'].tolist()))
        return tokens, pos
    elif dataset_idx == 0:
        tokens = sample['tokens']
        pos = list(range(len(sample['tokens'])))
        return tokens, pos
    elif dataset_idx == 3:
        tokens = sample['tokens']
        pos = list(
                set(
                    torch.cat([
                        sample['is_title|probe_indices'], sample['start_math|probe_indices'], sample['is_display_math|probe_indices'],
                        sample['is_math|probe_indices'], sample['is_frac|probe_indices'], sample['end_math|probe_indices'], 
                        sample['is_superscript|probe_indices'], sample['is_abstract|probe_indices'], sample['is_inline_math|probe_indices'],
                        sample['is_subscript|probe_indices'], sample['is_denominator|probe_indices'], sample['is_reference|probe_indices'],
                        sample['is_author|probe_indices'], sample['is_numerator|probe_indices']]
                    ).tolist()
                )
        )
        return tokens, pos
    elif dataset_idx == 2:
        tokens = sample['tokens']
        pos = list(
                set(
                    torch.cat([
                        sample['dep_root|probe_indices'], sample['upos_PROPN|probe_indices'], sample['VerbForm_Ger|probe_indices'],
                        sample['Mood_Imp|probe_indices'], sample['Gender_Fem|probe_indices'], sample['dep_conj|probe_indices'],
                        sample['PronType_Rel|probe_indices'], sample['dep_obj|probe_indices'], sample['VerbForm_Fin|probe_indices'],
                        sample['dep_nsubj:pass|probe_indices'], sample['upos_SYM|probe_indices'], sample['upos_PUNC|probe_indices'],
                        sample['PronType_Int|probe_indices'], sample['dep_parataxis|probe_indices'], sample['upos_ADJ|probe_indices'],
                        sample['Number_Plur|probe_indices'], sample['dep_advmod|probe_indices'], sample['dep_case|probe_indices'],
                        sample['Tense_Past|probe_indices'], sample['dep_aux|probe_indices'], sample['PronType_Prs|probe_indices'],
                        sample['Gender_Neut|probe_indices'], sample['first_eos_True|probe_indices'], sample['upos_AUX|probe_indices'],
                        sample['Person_1|probe_indices'], sample['dep_flat|probe_indices'], sample['dep_advcl|probe_indices'],
                        sample['NumType_Card|probe_indices'], sample['upos_NUM|probe_indices'], sample['upos_ADV|probe_indices'],
                        sample['dep_acl:relcl|probe_indices'], sample['dep_nmod|probe_indices'], sample['upos_DET|probe_indices'],
                        sample['VerbForm_Part|probe_indices'], sample['Voice_Pass|probe_indices'], sample['dep_obl|probe_indices'],
                        sample['upos_VERB|probe_indices'], sample['dep_xcomp|probe_indices'], sample['PronType_Art|probe_indices'],
                        sample['dep_nummod|probe_indices'], sample['upos_ADP|probe_indices'], sample['Person_2|probe_indices'],
                        sample['upos_SCONJ|probe_indices'], sample['dep_det|probe_indices'], sample['PronType_Dem|probe_indices'],
                        sample['dep_cc|probe_indices'], sample['dep_compound|probe_indices'], sample['upos_X|probe_indices'],
                        sample['upos_NOUN|probe_indices'], sample['dep_appos|probe_indices'], sample['dep_cop|probe_indices'],
                        sample['dep_list|probe_indices'], sample['upos_INTJ|probe_indices'], sample['upos_CCONJ|probe_indices'],
                        sample['upos_PRON|probe_indices'], sample['dep_aux:pass|probe_indices'], sample['Person_3|probe_indices'],
                        sample['dep_nsubj|probe_indices'], sample['eos_True|probe_indices'], sample['VerbForm_Inf|probe_indices'],
                        sample['dep_acl|probe_indices'], sample['dep_mark|probe_indices'], sample['Gender_Masc|probe_indices'],
                        sample['dep_amod|probe_indices'], sample['dep_ccomp|probe_indices'], sample['dep_punct|probe_indices'],
                        sample['dep_nmod:poss|probe_indices']]
                    ).tolist()
                )
        )
        return tokens, pos
    else:
        raise NotImplementedError(f"Preprocessing for {dataset_idx} is not implemented")
        
        

    