# Generating Captions for Instagram Food Images

This repository was based off the image captioning pytorch [tutorial](https://github.com/yunjey/pytorch-tutorial) developed by user [yunjey](https://github.com/yunjey).

It has been modified for generating image captions for Instagram photos, using embeddings from ResNet or a GAN, trained separately.


### To train the RNN->RNN model with default parameters, run:
`python train.py --pretrain_rnn --model_name NAME`

### To train the CNN->RNN model with default parameters, run:
`python train.py --model_name NAME`

### To train the DCGAN->RNN model with default parameters, run:
`python train.py --gan_embedding --model_name NAME`

### To train the ProGAN->RNN model with default parameters, run:
`python train.py --gan_embedding --model_name NAME`

### To generate captions for a dataset, run:
`python train.py --eval_only --print_cap_step 1 --log_step 1 --model_name NAME --resume MODEL_TO_EVALUATE --test_image_path PATH --test_caption_path PATH`

```
usage: train.py [-h] --model_name MODEL_NAME [--model_path MODEL_PATH]
                [--crop_size CROP_SIZE] [--vocab_path VOCAB_PATH]
                [--image_path IMAGE_PATH] [--caption_path CAPTION_PATH]
                [--log_step LOG_STEP] [--save_step SAVE_STEP]
                [--print_cap_step PRINT_CAP_STEP] [--resume RESUME]
                [--pretrain_rnn]
                [--pretrain_caption_path PRETRAIN_CAPTION_PATH]
                [--beam_search] [--multiple_datasets]
                [--embed_size EMBED_SIZE]
                [--encoder_rnn_hidden_size ENCODER_RNN_HIDDEN_SIZE]
                [--decoder_rnn_hidden_size DECODER_RNN_HIDDEN_SIZE]
                [--num_layers NUM_LAYERS] [--gan_embedding]
                [--progan_embedding] [--num_epochs NUM_EPOCHS]
                [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
                [--learning_rate LEARNING_RATE] [--eval_only]
                [--test_image_path TEST_IMAGE_PATH]
                [--test_caption_path TEST_CAPTION_PATH] [--make_cap_files]

optional arguments:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        model's identifying name, for notekeeping
  --model_path MODEL_PATH
                        path for saving trained models
  --crop_size CROP_SIZE
                        size for randomly cropping images
  --vocab_path VOCAB_PATH
                        path for vocabulary wrapper
  --image_path IMAGE_PATH
                        path for resized images
  --caption_path CAPTION_PATH
                        path for train captions
  --log_step LOG_STEP   step size for prining log info
  --save_step SAVE_STEP
                        step size for saving trained models
  --print_cap_step PRINT_CAP_STEP
                        step size for printing captions
  --resume RESUME       resume model training from most recent checkpoint for
                        this model name
  --pretrain_rnn        train an rnn->rnn model to improve the decoderRNN's
                        performance
  --pretrain_caption_path PRETRAIN_CAPTION_PATH
                        integer preprocessed captions for pretraining the rnn
                        decoder
  --beam_search         use beam_search instead of multinomial sampling
                        decoding
  --multiple_datasets   use multiple image/caption dataset files
  --embed_size EMBED_SIZE
                        dimension of word embedding vectors
  --encoder_rnn_hidden_size ENCODER_RNN_HIDDEN_SIZE
                        dimension of encoder hidden states
  --decoder_rnn_hidden_size DECODER_RNN_HIDDEN_SIZE
                        dimension of decoder hidden states
  --num_layers NUM_LAYERS
                        number of layers in lstm
  --gan_embedding       use a trained GAN to provide image embeddings for the
                        RNN. Use ResNet otherwise.
  --progan_embedding    use a trained proGAN to provide image embeddings for
                        the RNN. Use ResNet otherwise.
  --num_epochs NUM_EPOCHS
  --batch_size BATCH_SIZE
  --num_workers NUM_WORKERS
  --learning_rate LEARNING_RATE
  --eval_only           evaluate the specified model on the test dataset
  --test_image_path TEST_IMAGE_PATH
                        path for test images
  --test_caption_path TEST_CAPTION_PATH
                        path for train captions
  --make_cap_files      when trying to caption images, it makes one file per
                        image, with many captions in that file.
```


# README From Yunjey's original tutorial
## Image Captioning
The goal of image captioning is to convert a given input image into a natural language description. The encoder-decoder framework is widely used for this task. The image encoder is a convolutional neural network (CNN). In this tutorial, we used [resnet-152](https://arxiv.org/abs/1512.03385) model pretrained on the [ILSVRC-2012-CLS](http://www.image-net.org/challenges/LSVRC/2012/) image classification dataset. The decoder is a long short-term memory (LSTM) network. 

![alt text](png/model.png)

#### Training phase
For the encoder part, the pretrained CNN extracts the feature vector from a given input image. The feature vector is linearly transformed to have the same dimension as the input dimension of the LSTM network. For the decoder part, source and target texts are predefined. For example, if the image description is **"Giraffes standing next to each other"**, the source sequence is a list containing **['\<start\>', 'Giraffes', 'standing', 'next', 'to', 'each', 'other']** and the target sequence is a list containing **['Giraffes', 'standing', 'next', 'to', 'each', 'other', '\<end\>']**. Using these source and target sequences and the feature vector, the LSTM decoder is trained as a language model conditioned on the feature vector.

#### Test phase
In the test phase, the encoder part is almost same as the training phase. The only difference is that batchnorm layer uses moving average and variance instead of mini-batch statistics. This can be easily implemented using [encoder.eval()](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/sample.py#L37). For the decoder part, there is a significant difference between the training phase and the test phase. In the test phase, the LSTM decoder can't see the image description. To deal with this problem, the LSTM decoder feeds back the previosly generated word to the next input. This can be implemented using a [for-loop](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py#L48).



## Usage 


#### 1. Clone the repositories
```bash
$ git clone https://github.com/pdollar/coco.git
$ cd coco/PythonAPI/
$ make
$ python setup.py build
$ python setup.py install
$ cd ../../
$ git clone https://github.com/yunjey/pytorch-tutorial.git
$ cd pytorch-tutorial/tutorials/03-advanced/image_captioning/
```

#### 2. Download the dataset

```bash
$ pip install -r requirements.txt
$ chmod +x download.sh
$ ./download.sh
```

#### 3. Preprocessing

```bash
$ python build_vocab.py   
$ python resize.py
```

#### 4. Train the model

```bash
$ python train.py    
```

#### 5. Test the model 

```bash
$ python sample.py --image='png/example.png'
```

<br>

## Pretrained model
If you do not want to train the model from scratch, you can use a pretrained model. You can download the pretrained model [here](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and the vocabulary file [here](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0). You should extract pretrained_model.zip to `./models/` and vocab.pkl to `./data/` using `unzip` command.
