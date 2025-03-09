cd /home/mltrain/delta-selectivity-saes
#CUDA_LAUNCH_BLOCKING=1
python activations.py --model_name "EleutherAI/pythia-160m" --sae_name "EleutherAI/sae-pythia-160m-32k" --dataset_idx 11 --device cuda:0 --layers 0,1,2,3,4,5,6,7,8,9,10,11 --output_directory activations_pythia_160m/wikidata_gender
