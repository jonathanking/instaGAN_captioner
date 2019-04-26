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
from model import EncoderRNN, EncoderCNN, DecoderRNN, DecoderRNNOld
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import torchvision
from numpy import random
from load_model import DCGAN, Discriminator, Generator

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
    else:
        encoder = EncoderCNN(args.embed_size).to(device)

    decoder = DecoderRNNOld(args.embed_size, args.decoder_rnn_hidden_size, len(vocab), args.num_layers, vocab).to(device)
    return encoder, decoder


def load_model_weights(encoder, decoder):
    """ Loads weights for encoder and decoder. Also returns epoch and iteration to resume at."""
    if args.pretrain_rnn:
        encoder_file = sorted(glob("models/pretrain_rnn_encoder*.ckpt"))[-1]
        print("Loading", encoder_file)
        encoder.load_state_dict(torch.load(encoder_file))
    elif not args.gan_embedding and len(glob("models/encoder*.ckpt")) > 0:
        encoder_file = sorted(glob("models/encoder*.ckpt"))[-1]
        print("Loading", encoder_file)
        encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = sorted(glob("models/decoder*.ckpt"))[-1]
    print("Loading", decoder_file)
    decoder.load_state_dict(torch.load(decoder_file))
    starting_i = int(decoder_file.split("-")[2][:-5])
    starting_epoch = int(decoder_file.split("-")[1])
    return encoder, decoder, starting_epoch, starting_i


def get_trainable_params(encoder, decoder):
    """ Returns the parameters to train on based on the training configuration. """
    params = list(decoder.parameters())
    if args.pretrain_rnn:
        params += list(encoder.parameters())
    elif not args.gan_embedding:
        params += list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    return params


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])\

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    if args.pretrain_rnn:
        data_loader = get_caption_loader(args.pretrain_caption_path, vocab, args.batch_size, shuffle=True,
                                         num_workers=args.num_workers)
    else:
        data_loader = get_loader(args.caption_path, args.image_path,  vocab, transform, args.batch_size, shuffle=True,
                                 num_workers=args.num_workers)

    # Build the models
    encoder, decoder = get_encoder_decoder(vocab)

    # Resume model training if requested
    starting_epoch, starting_i = 0, 0
    if args.resume and len(glob("models/*.ckpt")) > 0:
        encoder, decoder, starting_epoch, starting_i = load_model_weights(encoder, decoder)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = get_trainable_params(encoder, decoder)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    logfile = open("logs/model.train", "w", 1)
    logfile.write(str(args) + "\n")
    logger = csv.writer(logfile)

    # Train the models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        # src = {images, captions}, tgt = {captions, captions[1:]}
        for i, (src_data, tgt_data, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            targets = pack_padded_sequence(tgt_data, lengths, batch_first=True)[0]

            # loss = train_step(images, captions, gan.discriminator, decoder, optimizer, criterion)

            # Forward, backward and optimize
            if args.gan_embedding:
                disc, features = encoder(src_data)
            else:
                features = encoder(src_data)
            outputs = decoder(features, tgt_data, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            if not args.gan_embedding:
                encoder.zero_grad()
            loss.backward()
            optimizer.step()

            end_of_epoch_cleanup(i, epoch, total_step, loss, tgt_data, vocab, encoder, decoder, starting_i,
                                 starting_epoch, src_data, features, lengths, logger)
    logfile.close()


def end_of_epoch_cleanup(i, epoch, total_step, loss, tgt_data, vocab, encoder, decoder, starting_i, starting_epoch,
                         src_data, features, lengths, logger):
    """ Execute bookkeeping methods such as printing the current epoch loss, example captions, and saving checkpts. """
    # Print log info
    if i % args.log_step == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
        logger.writerow([epoch, i, loss.item(), np.exp(loss.item())])

    if (epoch == 0 and i >= 4000 and i % args.print_cap_step == 0) or (epoch > 0 and i % args.print_cap_step == 0):
        with torch.no_grad():
            # for idx in range(args.batch_size):
            idx = 0
            tgt_caption = get_caption_from_tensor(tgt_data[idx][:lengths[idx]].cpu().numpy(), vocab)
            if args.pretrain_rnn:
                pred_caption, ll = decoder.beam_search_decode(features[idx], starting_char=tgt_data[idx][0])
            else:
                pred_caption, ll = decoder.beam_search_decode(features[idx])
            print("Target caption:", tgt_caption)
            print("Predicted caption:", get_caption_from_tensor(pred_caption, vocab))
            print("log-likelihood:", ll.item() / len(pred_caption))
            if not args.pretrain_rnn:  # aka, 'if source is an image'
                torchvision.utils.save_image(src_data[idx], "images/i{0}_{1}.png".format(i, idx))

    # Save the model checkpoints
    if (i + 1) % args.save_step == 0:
        torch.save(decoder.state_dict(), os.path.join(
            args.model_path, 'decoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))
        if args.pretrain_rnn:
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'pretrain_rnn_encoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))
        elif not args.gan_embedding:
            torch.save(encoder.state_dict(), os.path.join(
                args.model_path, 'encoder-{:08}-{:08}.ckpt'.format(starting_epoch + epoch + 1, starting_i + i + 1)))


def train_step(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=200):
    """ Executes the encoding and decoding of the image. """
    teacher_forcing_ratio = 0.5
    target_length = target_tensor.size(0)
    loss = 0

    _, encoder_hidden = encoder(input_tensor) # aka embedded_img = encoder(images)

    decoder_input = torch.zeros(1, 1, device='cuda', dtype=torch.long)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == END:
                break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


def get_caption_from_tensor(caption_tensor, vocab):
    """ Decodes a integer tensor caption using a defined vocabulary. """
    caption = "".join(map(vocab.decode, caption_tensor))
    return caption


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_path', type=str, default='data/train_data_1_eng.tch', help='path for resized images')
    parser.add_argument('--caption_path', type=str, default='data/train_meta_1_eng.pkl', help='path for train captions')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--print_cap_step', type=int, default=200, help='step size for printing captions')
    parser.add_argument('--resume', action="store_true", help='resume model training from most recent checkpoint')
    parser.add_argument('--pretrain_rnn', action="store_true",
                        help='train an rnn->rnn model to improve the decoderRNN\'s performance')
    parser.add_argument('--pretrain_caption_path', type=str, default='data/captions_en5_preprocessed.pt',
                        help='integer preprocessed captions for pretraining the rnn decoder')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=1024, help='dimension of word embedding vectors')
    parser.add_argument('--encoder_rnn_hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--decoder_rnn_hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=4, help='number of layers in lstm')
    parser.add_argument('--gan_embedding', action="store_true",
                        help='use a trained GAN to provide image embeddings for the RNN. Use ResNet otherwise.')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()

    if args.gan_embedding:
        assert args.embed_size == 1024, "RNN embedding size must match GAN embedding size of 1024."
    print(args)
    main(args)
