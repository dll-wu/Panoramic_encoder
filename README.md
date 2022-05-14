# Panoramic Encoder
This code can be used to reproduce the results of [Panoramic-Encoder: A Fast and Accurate Response Selection Paradigm for Generation-Based Dialogue Systems](https://arxiv.org/abs/2106.01263) on PersonaChat, Ubuntu V1, Ubuntu V2, and Douban datasets.

## Dependencies
The code is implemented using python 3.8 and PyTorch v1.8.1(please choose the correct command that match your CUDA version from [PyTorch](https://pytorch.org/get-started/previous-versions/))

Anaconda / Miniconda is recommended to set up this codebase.

There are only minor differences between the model files named "panoramic_encoder.py" in different directories, e.g., the pre-trained model used for the Douban corpus is "Bert-base-Chinese", not the English version.

You may use the command below:
```shell
conda create -n panoramic python=3.8

conda activate panoramic
cd Panoramic_encoder

pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Inference 

- We have uploaded standardized data and model checkpoints to Dropbox anonymously.

### PersonChat

- First, please download the [processed data and model checkpoint (421M in total)](https://www.dropbox.com/s/zx4bhdc83qd514y/data_and_checkpoint_for_PersonaChat.zip?dl=0).
- Inference
    ```shell
    unzip data_and_checkpoint_for_PersonaChat.zip -d persona-chat/
    cd persona-chat/
    python test.py --model_checkpoint checkpoint_for_PersonaChat

    ```

### Ubuntu V1 and V2

- First, please download the [processed data and model checkpoints (1.8G in total)](https://www.dropbox.com/s/nsr9otc7lrbu57x/data_and_checkpoint_for_Ubuntu.zip?dl=0).
- The post-trained baseline is from [BERT-FP(Han et al.)](https://github.com/hanjanghoon/BERT_FP).
- Inference
    ```shell
    unzip data_and_checkpoint_for_Ubuntu.zip -d ubuntu/
    cd ubuntu/

    # inference on Ubuntu V1
    python test.py --data_path data/UbuntuV1_data.json --model_checkpoint checkpoint_for_UbuntuV1

    # inference on Ubuntu V2
    python test.py --data_path data/UbuntuV2_data.json --model_checkpoint checkpoint_for_UbuntuV2

    # inference on Ubuntu V1(finetune from the checkpoint given by BERT-FP)
    python test.py --data_path data/UbuntuV1_data.json --use_post_training bert_fp --model_checkpoint checkpoint_for_UbuntuV1_use_bert_FP
    ```

### Douban

- First, please download the [processed data and model checkpoint (892M in total)](https://www.dropbox.com/s/qp4b6r8a32d21rt/data_and_checkpoint_for_Douban.zip?dl=0).
- Inference
    ```shell
    unzip data_and_checkpoint_for_Douban.zip -d douban/
    cd douban/

    # inference on Douban
    python test.py --model_checkpoint checkpoint_for_Douban

    # inference on Douban(finetune from the checkpoint given by BERT-FP)
    python test.py --use_post_training bert_fp --model_checkpoint checkpoint_for_Douban_use_bert_FP
    ```
