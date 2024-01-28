#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=generate-mdm
#SBATCH --mail-user=user@email.com #TODO: fill in your correct email
#SBATCH --mail-type="ALL"
#SBATCH --time=00:40:00
#SBATCH --partition=gpu-short
#SBATCH --output=/home/motion-diffusion-model/save/logs/%x_%j.out  #TODO: fill in yourß correct location
#SBATCH --gres=gpu:1

echo "## Starting MDM generating on $HOSTNAME at $(pwd)"

source $HOME/data1/thesis-env/bin/activate      #TODO: load your own env and modules you need. 
echo "## using python:"
which python
export PYTHONPATH=".":$PYTHONPATH



model=$HOME/motion-diffusion-model/save/swdance/model000535000.pt
output_dir=$HOME/motion-diffusion-model/save/swdance/samples

# ------ GENERATE FROM TEXTFILE ------ # 
textfile=$HOME/motion-diffusion-model/sample/val_texts.txt        

echo "Generating for file: $textfile and model: $model"
python -m sample.generate \
    --model_path $model\
    --input_text $textfile \
    # --num_repetitions 1 \
    --output_dir $output_dir \
    # --freeze_layers 3 \
    # --motion_length 2 # in seconds


# ------- GENERATE FROM BELOW DEFINED PROMPTS ------ # 
# declare -a prompts=(
#     "When do we draw the line? joy delight happiness passion excitement love"
#     "When do we draw the line?"
# )

# for prompt in "${prompts[@]}"
# do
#     echo "Generating for prompt: $prompt"
#     python -m sample.generate \
#         --model_path $model \
#         --text_prompt "$prompt" \
#         --num_repetitions 1 \
#         --output_dir $output_dir 

#     echo "done with prompt $prompt"
# done

# echo "done generating!" 




