cd /home/mltrain/delta-selectivity-saes
python activations.py --model_name "EleutherAI/pythia-410m" --sae_name "EleutherAI/sae-pythia-410m-65k" --dataset_idx 6 --device cuda:0 --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --output_directory activations_pythia_410m/text_features --model_hook_template "blocks.<layer>.hook_mlp_out" --sae_layer_template "layers.<layer>.mlp"
