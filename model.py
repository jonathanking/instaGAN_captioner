import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np

START = "\\"
END = "\n"


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """Represents a simple RNN for generating an embedding of the entire input sequence"""
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, inputs):
        """Extract feature vectors from input sequence."""
        embedded = self.embedding(inputs.long())
        output, hidden = self.rnn(embedded)
        return output[:,-1,:].squeeze()
        # return output, hidden


class EncoderCNN(nn.Module):
    def __init__(self, hidden_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device='cuda')


class DecoderRNNOld(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, vocab, max_seq_length=400):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNNOld, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        outputs_lengths, hidden_state = self.gru(packed)
        outputs = outputs_lengths[0]
        outputs = self.output_layer(outputs)
        return outputs

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.gru(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.output_layer(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids

    def decode(self, prev_hidden: torch.tensor, input: int) -> (torch.tensor, torch.tensor):
        """ Run the decoder AND the output layer for a single step.
        :param prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                            in the assignment
        :param input: int - the word being inputted to the decoder.
                                during log-likelihood computation, this is y_{t-1}
                                during greedy decoding, this is yhat_{t-1}
        :param model: a Seq2Seq model
        :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
                 (2) a tensor `hidden` of shape [L, hidden_dim], denoted H^{dec}_t in the assignment
        """

        def embed_tgt_word(w):
            """ Given an int representing a source vocab word, returns an embedding of this word."""
            word_emb = self.embed(w.long())
            return word_emb

        outputs, new_hidden_state = self.gru.forward(embed_tgt_word(input).view(1,1,-1), prev_hidden)
        probabilities = F.softmax(self.output_layer.forward(new_hidden_state[-1]))
        return probabilities, new_hidden_state

    def select_top_k(self, candidates, k):
        """ Given a list of candidates, each a tuple of (sentence, log-likelihood, model-state), returns the top k
            candidates according to length normalized log-likelihood. """
        candidate_nlls = list(map(lambda x: x[1]/len(x[0]), candidates))
        candidate_idxs = list(range(len(candidates)))
        top_k_idxs = candidate_idxs[:k]
        top_k_nlls = candidate_nlls[:k]
        worst_val = min(top_k_nlls)
        worst_idx = top_k_nlls.index(worst_val)
        for i, cnll in enumerate(candidate_nlls[k:]):
            if cnll > worst_val:
                top_k_idxs[worst_idx] = i + k
                top_k_nlls[worst_idx] = cnll
                worst_val = min(top_k_nlls)
                worst_idx = top_k_nlls.index(worst_val)
        return [candidates[j] for j in top_k_idxs]

    def beam_search_decode(self, encodings, beam_width=4, max_length=150, starting_char=None, randomize_prob=False):
        if not starting_char:
            starting_char = self.vocab(START)
        zero_state = torch.zeros(self.num_layers, 1, self.hidden_size, device="cuda")
        outputs, prev_hs = self.gru.forward(encodings.view(1,1,-1), zero_state)
        probs, prev_hs = self.decode(prev_hs, torch.Tensor([starting_char]).cuda())
        first_model_state = (probs, prev_hs)
        candidate_translations = [([], 0, first_model_state)]
        final_candidate_translations = []

        def add_noise(p):
            """ Adds Gaussian noise to the probability to diversify outputs."""
            random_prob = p - np.random.normal(0, 0.001)
            if random_prob > 1:
                random_prob = torch.tensor(1, dtype=torch.float, device="cuda")
            elif random_prob < 0:
                random_prob = torch.tensor(0, dtype=torch.float, device="cuda")
            return random_prob

        def expand_candidates(candidates):
            expanded_by_one = []
            for i, (h, old_ll, model_state) in enumerate(candidates):
                probs, prev_hs = model_state
                for candidate_word in range(1, self.vocab_size):
                    candidate_sentence = h + [candidate_word]
                    prob = probs[0][candidate_word]
                    if randomize_prob:
                        prob = add_noise(prob)
                    new_ll = torch.log(prob)
                    ll = old_ll + new_ll
                    expanded_by_one.append((candidate_sentence, ll, model_state))
            return expanded_by_one

        def prune_candidates(candidates):
            candidates = self.select_top_k(candidates, beam_width)
            candidates = candidates[-beam_width:]
            f = []
            for c, ll, ms in candidates:
                probs, prev_hs = ms
                probs, prev_hs = self.decode(prev_hs, torch.tensor(c[-1], dtype=torch.long, device='cuda'))
                ms = probs, prev_hs
                f.append((c, ll, ms))

            return f

        while beam_width > 0:
            expanded_candidates = expand_candidates(candidate_translations)
            # print("expanded", len(expanded_candidates), [x[0] for x in expanded_candidates[:4]])
            candidate_translations = prune_candidates(expanded_candidates)
            # print("pruned", len(candidate_translations), [(words(x[0]), x[1].item()/len(x[0])) for x in candidate_translations[:4]])
            if any(map(lambda c_ll_ms: len(c_ll_ms[0]) >= max_length, candidate_translations)):
                final_candidate_translations += [(c, ll) for (c, ll, ms) in candidate_translations]
                break

            finished_translations = [(c, ll) for (c, ll, ms) in candidate_translations if (c[-1] == self.vocab(END))]
            if len(finished_translations) > 0:
                final_candidate_translations.extend(finished_translations)
                beam_width -= len(finished_translations)
            candidate_translations = [(c, ll, ms) for (c, ll, ms) in candidate_translations if not (c[-1] == self.vocab(END))]

        best_translation = None
        best_ll = None
        for ft, ll in final_candidate_translations:
            if best_ll is None:
                best_translation = ft
                best_ll = ll
            elif ll > best_ll:
                best_ll = ll
                best_translation = ft

        return best_translation, best_ll


def generate_text(decoder, encodings, vocab, starting_char=START, temp=1.4, decrease_temp=False):
    """ Given a decoder and a tensor provided by the model's encoder representing the image or caption to generate
        a caption from, this function users the decoder to sample from a multinomial distribution of next character
        probabilities. """
    # Number of characters to generate
    num_generate = 1000
    difference = temp - 1
    if decrease_temp:
        change = difference / num_generate
    else:
        change = 0

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    temperature = temp

    # Initialize decoder with encodings
    zero_state = torch.zeros(decoder.num_layers, 1, decoder.hidden_size, device="cuda")
    outputs, prev_hs = decoder.gru.forward(encodings.view(1, 1, -1), zero_state)
    probs, prev_hs = decoder.decode(prev_hs, torch.tensor([vocab(starting_char)], device="cuda"))

    ll = torch.tensor(0, dtype=torch.float)

    for i in range(num_generate):
        # remove the batch dimension
        probs = probs.squeeze()

        # using a multinomial distribution to predict the word returned by the model
        probs_temp = probs / temperature
        char = torch.multinomial(probs_temp, 1)
        ll += probs[char.item()]
        temperature -= change

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        probs, prev_hs = decoder.decode(prev_hs, torch.tensor([char], device="cuda"))

        text_generated.append(char.item())
        if char.item() == vocab("\n"):
            break

    return text_generated, ll