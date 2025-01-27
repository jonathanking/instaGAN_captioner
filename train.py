# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import os
import pickle
import csv
from data_loader import get_loader, get_caption_loader
from build_vocab import Vocabulary
from model import EncoderRNN, EncoderCNN, DecoderRNN, DecoderRNNOld, generate_text, ProGANToRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torchvision
from numpy import random
from load_model import DCGAN, Discriminator, Generator
import pro_gan_pytorch.PRO_GAN as pg

START = 0
END = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_encoder_decoder(vocab):
    """ Given the arguments, returns the correct combination of CNN/RNN/GAN encoders and decoders. """
    if args.pretrain_rnn:
        encoder = EncoderRNN(len(vocab), args.embed_size, args.encoder_rnn_hidden_size, num_layers=args.num_layers).to(device)
    elif args.gan_embedding:
        gan = torch.load('DCGAN_embed_2.tch').to(device)
        encoder = gan.discriminator
    elif args.progan_embedding:
        pro_gan = pg.ProGAN(depth=7, latent_size=256, device=torch.device('cuda'))
        pro_gan.dis.load_state_dict(torch.load('progan_weights/GAN_DIS_6.pth'))
        # pro_gan.dis_optim.load_state_dict(torch.load('progan_weights/GAN_DIS_OPTIM_6.pth'))
        pro_gan.gen.load_state_dict(torch.load('progan_weights/GAN_GEN_6.pth'))
        # pro_gan.gen_optim.load_state_dict(torch.load('progan_weights/GAN_GEN_OPTIM_6.pth'))
        pro_gan.gen_shadow.load_state_dict(torch.load('progan_weights/GAN_GEN_SHADOW_6.pth'))
        print("Loaded proGAN weights.", flush=True)
        encoder = pro_gan.dis.to(device)
    else:
        encoder = EncoderCNN(args.embed_size).to(device)

    decoder = DecoderRNNOld(args.embed_size, args.decoder_rnn_hidden_size, len(vocab), args.num_layers, vocab, device=device).to(device)
    return encoder, decoder


def load_model_weights(encoder, decoder):
    """ Loads weights for encoder and decoder. Also returns epoch and iteration to resume at."""
    resume_path = "models/" + args.resume + "/"
    # the encoder is a RNN
    if args.pretrain_rnn:
        encoder_file = sorted(glob(resume_path + "pretrain_rnn_encoder*.ckpt"))[-1]
        print("Loading", encoder_file)
        encoder.load_state_dict(torch.load(encoder_file))
    # the encoder is a CNN and the CNN state has been saved before
    elif not args.gan_embedding and len(glob(resume_path + "encoder*.ckpt")) > 0:
        encoder_file = sorted(glob(resume_path + "encoder*.ckpt"))[-1]
        print("Loading", encoder_file)
        encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = sorted(glob(resume_path + "decoder*.ckpt"))[-1]
    print("Loading", decoder_file)
    decoder.load_state_dict(torch.load(decoder_file))
    starting_i = int(decoder_file.split("-")[2][:-5])
    starting_epoch = int(decoder_file.split("-")[1])
    return encoder, decoder, starting_epoch, starting_i


def get_trainable_params(encoder, decoder, enc2dec_transformation=None):
    """ Returns the parameters to train on based on the training configuration. """
    params = list(decoder.parameters())
    if args.progan_embedding and enc2dec_transformation:
        params += list(enc2dec_transformation.parameters())
    if args.pretrain_rnn:
        params += list(encoder.parameters())
    elif not args.gan_embedding and not args.progan_embedding:
        params += list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    return params


def main():
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Initialize logging mechanisms
    logdir = "logs/" + args.model_name + "/"
    imgdir = logdir + "imgs/"
    chkptdir = "models/" + args.model_name + "/"
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(chkptdir, exist_ok=True)
    with open(logdir + args.model_name + ".args", "w") as f:
        f.write(str(args) + "\n")
    logfile = open(logdir + args.model_name + ".train", "w", 1)
    logheader = "epoch,iter,loss,perplexity,pred_caption,targ_caption,caption_ll,caption_perplexity,temp\n"
    logfile.write(logheader)
    logger = csv.writer(logfile, quotechar='“', quoting=csv.QUOTE_NONNUMERIC)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])\

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build the models
    encoder, decoder = get_encoder_decoder(vocab)
    if args.progan_embedding:
        linear_transformation = ProGANToRNN(args.embed_size).to(device)
    else:
        linear_transformation = None

    # Resume model training if requested
    starting_epoch, starting_i = 0, 0
    if args.resume and len(glob("models/" + args.resume + "/" + "*.ckpt")) > 0:
        encoder, decoder, starting_epoch, starting_i = load_model_weights(encoder, decoder)
    print("Models built.", flush=True)

    # Build data loader
    if args.pretrain_rnn:
        data_loader = get_caption_loader(args.pretrain_caption_path, vocab, args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, seq_len=75)
    elif not args.eval_only:
        data_loader = get_loader(args.caption_path, args.image_path,  vocab, transform, args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, use_multiple_files=args.multiple_datasets)
    elif args.eval_only:
        data_loader = get_loader(args.test_caption_path, args.test_image_path,  vocab, transform, args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, use_multiple_files=args.multiple_datasets)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = get_trainable_params(encoder, decoder, None)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    if args.progan_embedding:
        linear_transformation_optimizer = torch.optim.Adam(linear_transformation.parameters(), lr=args.learning_rate*100)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        # src = {images, captions}, tgt = {captions, captions[1:]}
        for i, (src_data, tgt_data, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            targets = pack_padded_sequence(tgt_data, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            if args.gan_embedding:
                disc, features = encoder(src_data)
            elif args.progan_embedding:
                # here, features is a 4096 len vector from the proGAN model's discriminator
                disc, features = encoder(x=src_data, height=6, alpha=0.5, output_embeddings=True)
                features = linear_transformation(features)
            else:
                features = encoder(src_data)
            outputs = decoder(features, tgt_data, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            if not args.gan_embedding:
                encoder.zero_grad()
            if not args.eval_only:
                loss.backward()
                optimizer.step()
                if args.progan_embedding:
                    linear_transformation_optimizer.step()

            end_of_epoch_cleanup(i, epoch, total_step, loss, tgt_data, vocab, encoder, decoder, starting_i,
                                 starting_epoch, src_data, features, lengths, logger, imgdir)
    logfile.close()


def end_of_epoch_cleanup(i, epoch, total_step, loss, tgt_data, vocab, encoder, decoder, starting_i, starting_epoch,
                         src_data, features, lengths, logger, imgdir):
    """ Execute bookkeeping methods such as printing the current epoch loss, example captions, and saving checkpts. """
    pred_caption_str = ""
    tgt_caption_str = ""
    t = 0
    ll = 0
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255])


    if (epoch == 0 and i >= 0 and i % args.print_cap_step == 0) or (epoch > 0 and i % args.print_cap_step == 0):
        with torch.no_grad():
            idx = 0
            tgt_caption = get_caption_from_tensor(tgt_data[idx][:lengths[idx]].cpu().numpy(), vocab)
            t = np.exp(np.random.normal(.3, 0.2))
            if args.pretrain_rnn and args.beam_search:
                pred_caption, ll = decoder.beam_search_decode(features[idx], starting_char=tgt_data[idx][0])
            elif args.pretrain_rnn and not args.beam_search:
                pred_caption, ll = generate_text(decoder, features[idx], vocab, temp=t, starting_char=tgt_data[idx][0])
            elif not args.pretrain_rnn and args.beam_search:
                pred_caption, ll = decoder.beam_search_decode(features[idx], randomize_prob=False)
            elif not args.pretrain_rnn and not args.beam_search:
                pred_caption, ll = generate_text(decoder, features[idx], vocab, temp=t)
            pred_caption_str = get_caption_from_tensor(pred_caption, vocab).replace("\n", "")
            tgt_caption_str = tgt_caption.strip().replace("\n", "")[1:]
            ll = ll.item()
            print("Target caption:", tgt_caption_str.strip())
            print("Predicted caption:", pred_caption_str.strip())
            print("Temp = {0:4f}".format(t))
            print("log-likelihood:", ll / len(pred_caption))
            if args.make_cap_files:
                with open("logs/final_captions/{}.txt".format(i), "a") as f:
                    f.write(pred_caption_str.strip().replace("\t", "\n") + "*"*30 + "\n")
            if not args.pretrain_rnn:  # aka, 'if source is an image'
                torchvision.utils.save_image(inv_normalize(src_data[idx].to(device)), imgdir + "{0}_{1:03}.png".format(epoch, i))

    # Print log info
    if i % args.log_step == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
        q = '“'
        logger.writerow([epoch, i, loss.item(), np.exp(loss.item()), pred_caption_str, tgt_caption_str, ll, np.exp(ll), t])

    # Save the model checkpoints
    if (i + 1) % args.save_step == 0:
        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, args.model_name, 'decoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))
        if args.pretrain_rnn:
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, args.model_name, 'pretrain_rnn_encoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))
        elif not args.gan_embedding:
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))


def get_caption_from_tensor(caption_tensor, vocab):
    """ Decodes a integer tensor caption using a defined vocabulary. """
    caption = "".join(map(vocab.decode, caption_tensor))
    return caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="model's identifying name, for notekeeping", required=True)
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_path', type=str, default='data/food_training_eng.tch', help='path for resized images')
    parser.add_argument('--caption_path', type=str, default='data/food_training_meta_eng.pkl', help='path for train captions')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--print_cap_step', type=int, default=100, help='step size for printing captions')
    parser.add_argument('--resume', type=str, help='resume model training from most recent checkpoint for this model name')
    parser.add_argument('--pretrain_rnn', action="store_true",
                        help='train an rnn->rnn model to improve the decoderRNN\'s performance')
    parser.add_argument('--pretrain_caption_path', type=str, default='data/captions_en5_preprocessed.pt',
                        help='integer preprocessed captions for pretraining the rnn decoder')
    parser.add_argument('--beam_search', action="store_true",
                        help='use beam_search instead of multinomial sampling decoding')
    parser.add_argument('--multiple_datasets', action="store_true", help="use multiple image/caption dataset files")

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=512, help='dimension of word embedding vectors')
    parser.add_argument('--encoder_rnn_hidden_size', type=int, default=512, help='dimension of encoder hidden states')
    parser.add_argument('--decoder_rnn_hidden_size', type=int, default=512, help='dimension of decoder hidden states')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in lstm')
    parser.add_argument('--gan_embedding', action="store_true",
                        help='use a trained GAN to provide image embeddings for the RNN. Use ResNet otherwise.')
    parser.add_argument('--progan_embedding', action="store_true",
                        help='use a trained proGAN to provide image embeddings for the RNN. Use ResNet otherwise.')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--eval_only', action="store_true", help="evaluate the specified model on the test dataset")
    parser.add_argument('--test_image_path', type=str, default='data/food_training_6_eng.tch', help='path for test images')
    parser.add_argument('--test_caption_path', type=str, default='data/food_training_meta_6_eng.pkl',
                        help='path for train captions')
    parser.add_argument('--make_cap_files', action="store_true", help="when trying to caption images, it makes one file per image, with many captions in that file.")
    args = parser.parse_args()

    if args.gan_embedding:
        assert args.embed_size == 1024, "RNN embedding size must match GAN embedding size of 1024."
    print(args)
    main()
