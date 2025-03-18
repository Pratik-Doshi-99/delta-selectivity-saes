cd /home/mltrain/delta-selectivity-saes
#CUDA_LAUNCH_BLOCKING=1

python main.py --model_name EleutherAI/pythia-410m --sae_name EleutherAI/sae-pythia-410m-65k --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 --output_directory results_410m --batch_size 128 --output_size 1024 --datasets 7,8,9,10,11 --model_hook_template "blocks.<layer>.hook_mlp_out" --sae_layer_template "layers.<layer>.mlp" > wikidata_410m_logs 2>&1


python main.py --model_name EleutherAI/pythia-160m --sae_name EleutherAI/sae-pythia-160m-32k --layers 0,1,2,3,4,5,6,7,8,9,10,11 --output_directory results_160m --batch_size 512 --output_size 2048 --datasets 7,8,9,10,11 > wikidata_160m_logs 2>&1