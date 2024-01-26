# SWDance: Transfer learning $MDM$ to Generate Choreography from Spoken Word Input.  

The official PyTorch implementation of SWDance. 
*TODO: preview picture and visualisations.


## Getting started

This code is based on MDM (TODO insert link). Set up is according to their setup. 

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate mdm
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
```

Download dependencies:

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

### 2. Get data

Download [the SWDance dataset](https://drive.google.com/uc?export=download&id=1xNC1PBAMuCXSrUQxsm66oLdrHDNOLxi7 ), then unzip and place it in `./dataset/`. 

### 3. SWDance checkpoint

Download the [SWDance checkpoint]( https://drive.google.com/uc?export=download&id=1lAEDeSWyuvWWCMjJMXC6tDyp25tN0E8S), then unzip and place them in `./save/`. 


## Motion Synthesis

### Generate from test set prompts

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --num_samples 10 --num_repetitions 3
```

### Generate from your text file

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --input_text ./assets/example_text_prompts.txt
```

### Generate a single prompt

```shell
python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt "the person walked forward and is picking up his toolbox."
```

**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).

**Running those will get you:**

* `results.npy` file with text prompts and xyz positions of the generated animation
* `sample##_rep##.mp4` - a stick figure animation for each generated motion.

It will look something like this:

![example](assets/example_stick_fig.gif)

You can stop here, or render the SMPL mesh using MDM description.
MDM also allowds for motion editing, see their page for more details:
-  Unconditioned editin
-  Text conditioned editing

## Training

```shell
python -m train.train_mdm --save_dir save/my_swdance --dataset swdance
```

## Evaluation (TODO)


## Acknowledgments

This code is based on Motion Diffusion Model (TODO: link!). 

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
