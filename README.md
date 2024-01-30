# SWDance: Transfer learning MDM to Generate Choreography from Spoken Word Input.  

![frontcover](https://github.com/joycedh/MDM-SWDance/assets/33030971/3299eb52-5195-46f2-bc7e-9f768dfab909)

Submitted to the [9th International Conference on Movement and Computing](https://moco24.movementcomputing.org/). 

## Preparation

<details>
  <summary><b>Environment</b></summary>

We ran it on:
* Python 3.10
* CUDA capable GPU (one is enough)

```bash
pip install -r requirements.txt
```
</details>

<details>
  <summary><b>Dependencies</b></summary>

Prepare the following dependencies according to [MDM](https://github.com/GuyTevet/motion-diffusion-model/)

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

</details>

## Dataset

<details>
  <summary><b>The SWDance dataset</b></summary>

* Download the dataset files [here](https://drive.google.com/uc?export=download&id=1xNC1PBAMuCXSrUQxsm66oLdrHDNOLxi7 ), then unzip and place it in `./dataset/`. 
* Download the [SWDance checkpoint](https://drive.google.com/uc?export=download&id=1-NGpCRrBECuS6-6puxuSjYUmIhiRY-C7), then unzip and place it in `./save/`. 
</details>

<details>
  <summary><b>Your own dataset</b></summary>

Create your own text-to-dance dataset from a YouTube playlist, according to the instructions in `./dataset/swdance_dataset_pipeline.ipynb`. Place it in `./dataset/`, and make sure to also create and edit the correct `_opt.txt` files for it. 

</details> 

Make sure to run `train-val-split.ipynb` for correct train/val splitting of the dataset.


## Motion Synthesis

Based on [MDM](https://github.com/GuyTevet/motion-diffusion-model/)

<details>
  <summary><b>Generate from test set prompts</b></summary>

```shell
python -m sample.generate --model_path ./save/swdance/model000500000.pt --num_samples 10 --num_repetitions 3
```
</details> 

<details>
  <summary><b>Generate from your text file</b></summary>

```shell
python -m sample.generate --model_path ./save/swdance/model000500000.pt --input_text ./sample/val_texts.txt
```
</details> 


<details>
  <summary><b>Generate from a single prompt.</b></summary>

```shell
python -m sample.generate --model_path ./save/swdance/model000500000.pt --text_prompt "the person walked forward and is picking up his toolbox."
```
</details> 

**You may also define:**
* `--device` id.
* `--seed` to sample different prompts.
* `--motion_length` (text-to-motion only) in seconds (maximum is 9.8[sec]).
* `--freeze_layers` to specify the amount of frozen layers in your model. 

**Running those will get you:**

* `motions/prompt_title.npy` file with text prompts and xyz positions of the generated animation
* `prompt_title.gif` - a stick figure animation for each generated motion.

It will look something like the above banner. 


You can stop here, or render the SMPL mesh using a description provided in [MDM](https://github.com/GuyTevet/motion-diffusion-model/).
They also allowds for motion editing, see their page for more details on:
-  Unconditioned editin
-  Text conditioned editing 

## Training

```shell
python -m train.train_mdm --save_dir save/my_swdance --dataset swdance
```

Add `--freeze_layers` to specify how many layers of the checkpoint should be frozen during training. 

## Evaluation 

Evaluation is based on [Bailando](https://github.com/lisiyao21/Bailando). 
To run evaluation, see: `evaluation-bailando/evaluation_metrics.ipynb`.


## Acknowledgments

As this code is based on [MDM](https://github.com/GuyTevet/motion-diffusion-model/) and [Bailando](https://github.com/lisiyao21/Bailando), I would like to thank the creators of these repositories for their great minds and amazing work. 

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
