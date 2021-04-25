import math

import numpy as np
import torch
import torch.nn as nn

from .extra_layers import WeightDropout, RNNDropout, EmbeddingDropout


class GradReverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class AdversarialNetworkDANN(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetworkDANN, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class MultiLayerLSTM(nn.Module):
    """
    A multi-layer LSTM with specified output dimension and weight dropout.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=None,
                 bidirectional=False, dropout=0, weight_dropout=0, batch_first=True):
        super().__init__()
        self.num_layers = num_layers
        if output_size is None:
            output_size = hidden_size
        self.rnns = [nn.LSTM(input_size if l == 0 else hidden_size,
                             hidden_size if l != num_layers - 1 else output_size,
                             1, bidirectional=bidirectional,
                             batch_first=batch_first) for l in range(num_layers)]
        self.rnns = [WeightDropout(rnn, weight_dropout) for rnn in self.rnns]
        self.rnns = nn.ModuleList(self.rnns)
        self.hidden_dps = nn.ModuleList([RNNDropout(dropout) for l in range(num_layers)])

    def forward(self, inputs, hx=None, return_outputs=False):
        new_h = []
        raw_outputs = []
        drop_outputs = []
        outputs = inputs
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            outputs, hid = rnn(outputs, hx[l] if hx else None)
            new_h.append(hid)
            raw_outputs.append(outputs)
            if l != self.num_layers - 1:
                outputs = hid_dp(outputs)
            drop_outputs.append(outputs)
        if return_outputs:
            return outputs, new_h, raw_outputs, drop_outputs
        else:
            return outputs, new_h


class LSTMLanguageModel(nn.Module):
    #### FILE COPIED FROM fastai.text import awd_lstm.py
    #
    #   fastai.text   awd_lstm.py

    init_range = 0.1

    def __init__(self, vocab_size, emb_size=300, hidden_size=1150, num_layers=2,
                 output_p=0.4, hidden_p=0.3, input_p=0.4, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.vocab_size, self.emb_size, self.hidden_size, self.num_layers = \
            vocab_size, emb_size, hidden_size, num_layers
        self.output_p, self.hidden_p, self.input_p, self.embed_p, self.weight_p = \
            output_p, hidden_p, input_p, embed_p, weight_p
        self.bs = 1

        self.encoder = nn.Embedding(vocab_size, emb_size)
        self.encoder.weight.data.uniform_(-self.init_range, self.init_range)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.input_dp = RNNDropout(input_p)
        self.rnn = MultiLayerLSTM(emb_size, hidden_size, num_layers, emb_size,
                                  dropout=hidden_p, weight_dropout=weight_p)
        self.ouput_dp = RNNDropout(output_p)
        self.decoder = nn.Linear(emb_size, vocab_size)

        self.decoder.weight = self.encoder.weight
        self.reset()

    def forward(self, inputs):
        bs, sl = inputs.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()

        outputs = self.input_dp(self.encoder_dp(inputs))
        outputs, hidden, raw_outputs, drop_outputs = self.rnn(outputs, self.hidden, True)
        decoded = self.decoder(self.ouput_dp(outputs))
        self.hidden = self._to_detach(hidden)
        return decoded, raw_outputs, drop_outputs

    def _to_detach(self, x):
        if isinstance(x, (list, tuple)):
            return [self._to_detach(b) for b in x]
        else:
            return x.detach()

    def _init_h(self, l):
        n_hid = self.hidden_size if l != self.num_layers - 1 else self.emb_size
        return self.weights.new(1, self.bs, n_hid).zero_()

    def reset(self):
        self.weights = next(self.parameters()).data
        self.hidden = [(self._init_h(l), self._init_h(l)) for l in range(self.num_layers)]


class MultiDomainAdaptationModel(nn.Module):
    def __init__(self, n_langs, n_doms, vocab_sizes, num_layers=2, num_share=1):
        super().__init__()
        self.n_langs = n_langs
        self.n_doms = n_doms
        self.vocab_sizes = vocab_sizes
        self.num_layers = num_layers
        self.num_share = num_share
        self.alpha = 2
        self.beta = 1

        encoders = []
        models = []
        for lid in range(n_langs):
            for did in range(n_doms):
                lm = LSTMLanguageModel(vocab_sizes[lid])
                if lid > 0 or did > 0:  # share rnn layers across all langs / doms
                    for i in range(num_share):
                        lm.rnn.rnns[i] = models[0].rnn.rnns[i]
                if lid > 0:  # share domain specific rnn layers across languages
                    for i in range(num_share, num_layers):
                        lm.rnn.rnns[i] = models[did].rnn.rnns[i]
                if did == 0:
                    encoders.append(lm.encoder)
                else:
                    lm.encoder = models[-1].encoder
                    lm.encoder_dp = models[-1].encoder_dp

                lm.decoder.weight = lm.encoder.weight

                models.append(lm)

        self.models = nn.ModuleList(models)
        self.encoders = nn.ModuleList(encoders)
        self.crit = nn.CrossEntropyLoss()

    def encoder_weight(self, lid):
        return self.encoders[lid].weight.data.cpu().numpy()

    def _get_model_id(self, lid, did):
        return lid * self.n_doms + did

    def forward(self, inputs, lid, did):
        return self.models[self._get_model_id(lid, did)](inputs)

    def lm_loss(self, inputs, target, lid, did, return_h=False, return_d=False):
        decoded, raw_outputs, drop_outputs = self.models[self._get_model_id(lid, did)](inputs)
        loss = raw_loss = self.crit(decoded.view(-1, decoded.size(-1)), target.view(-1))
        if self.alpha > 0:  # AR regularization
            loss = loss + sum(self.alpha * h.pow(2).mean() for h in drop_outputs[-1:])
        if self.beta > 0:  # TAR regularization
            loss = loss + sum(self.beta * (h[:, 1:] - h[:, :-1]).pow(2).mean() for h in raw_outputs[-1:])
        if return_d:
            return raw_loss, loss, raw_outputs, decoded

        if return_h:
            return raw_loss, loss, raw_outputs


        else:
            return raw_loss, loss

    def cdan_loss(self, inputs, target, lid, did):
        decoded, raw_outputs, drop_outputs = self.models[self._get_model_id(lid, did)](inputs)

        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))

    def reset(self, lid=None, did=None):
        if lid is None or did is None:
            return [[self.reset(lid, did) for did in range(self.n_doms)] for lid in range(self.n_langs)]
        else:
            lm = self.models[self._get_model_id(lid, did)]
            hidden = lm.hidden
            lm.reset()
            return hidden

    def set_hidden(self, hidden, lid, did):
        mid = self._get_model_id(lid, did)
        self.models[mid].hidden = hidden


class MeanPoolClassifier(nn.Module):

    def __init__(self, input_size, n_classes, dropout):
        super().__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.dropout = dropout

        self.linear = nn.Linear(input_size, n_classes)
        self.dp = nn.Dropout(dropout)

    def forward(self, inputs, lengths=None):
        if not lengths:
            lengths= torch.tensor([min(len(xs),256) for xs in inputs]).cuda()
            # lengths.cuda()
        bs, sl, _ = inputs.size()
        idxes = torch.arange(0, sl).unsqueeze(0).to(inputs.device)
        mask = (idxes < lengths.unsqueeze(1)).float()
        pooled = (inputs * mask.unsqueeze(-1)).sum(1) / lengths.float().unsqueeze(-1)
        dropped = self.dp(pooled)
        # print(dropped[0].shape)
        return dropped, self.linear(dropped)


class SentimentClassifier(MultiDomainAdaptationModel):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

        self.clfs = [MeanPoolClassifier(input_size=300, n_classes=n_classes, dropout=0.6) for _ in range(self.n_doms)]
        self.clfs = nn.ModuleList(self.clfs)
        self.clf_crit = nn.CrossEntropyLoss()

    def forward(self, inputs, lengths, lid, did):
        prev_h = self.reset(lid, did)
        _, raw_outputs, _ = super().forward(inputs, lid, did)
        self.set_hidden(prev_h, lid, did)

        features, outputs = self.clfs[did](raw_outputs[-1], lengths)

        return features, outputs

    def clf_loss(self, inputs, lengths, label, lid, did):
        logits, _ = self.forward(inputs, lengths, lid, did)
        loss = self.clf_crit(logits, label)
        return loss

    def dm_loss(self, inputs, lengths, label, lid, did):
        outputs, features = self.forward(inputs, lengths, lid, did)
        softmax_out = nn.Softmax(dim=1)(outputs)

        entropy = Entropy(softmax_out)

        # transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)

    def cdan_smax(self, inputs, lengths,raw_outputs, lid, did):

        self.set_hidden(prev_h, lid, did)

        features, outputs = self.clfs[did](raw_outputs[-1], lengths)

        prev_h = self.reset(lid, did)
        _, raw_outputs, _ = super().forward(inputs, lid, did)
        self.set_hidden(prev_h, lid, did)
        return self.clfs[did](raw_outputs[-1], lengths)


class Discriminator(nn.Module):
    def __init__(self, input_size=300, hidden_size=400, output_size=2, num_layers=2, dropout=0.1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            layers.append(nn.Linear(n_in, n_out))
            if i < self.num_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_size=600, hidden_size=800, output_size=2, num_layers=2, dropout=0.1):
        super().__init__()

        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 50000.0


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.sigmoid = nn.Sigmoid()

        layers = []
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            layers.append(nn.Linear(n_in, n_out))
            if i < self.num_layers:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # if self.training:
        #     self.iter_num += 1
        # coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        #
        # x = x * 1.0
        # x.register_hook(grl_hook(coeff))
        x = self.layers(x)
        y = self.sigmoid(x)
        return y


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    # if random_layer is None:
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))

    op_out_new = GradReverse.apply(op_out.view(-1, softmax_output.size(1) * feature.size(1)))

    ad_out = ad_net(op_out_new)
    # else:
    #     random_out = random_layer.forward([feature, softmax_output])
    #     ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(
            weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
