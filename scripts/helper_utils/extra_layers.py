import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


#### FILE COPIED FROM fastai.text import WeightDropout, RNNDropout, EmbeddingDropout
#
#   fastai.text import WeightDropout, RNNDropout, EmbeddingDropout


def dropout_mask(x, sz, p: float):
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    def __init__(self, p: float=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class EmbeddingDropout(nn.Module):

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class WeightDropout(nn.Module):

    def __init__(self, module, weight_p, layer_names=['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            raw_w = nn.Parameter(w.data)
            self.register_parameter(f'{layer}_raw', raw_w)
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)
