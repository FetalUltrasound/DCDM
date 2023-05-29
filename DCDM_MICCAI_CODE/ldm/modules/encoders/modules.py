import torch
import torch.nn as nn
from functools import partial
# import clip
import timm
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding2D
#import kornia


from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError



class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=5, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        # print(c.shape)
        return c

class ReturnImage(nn.Module):
    def __init__(self, key='cond_image'):
        super().__init__()
        self.key = key


    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # print(batch)
        return batch[key]


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim==2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

class Dual_Embedder(nn.Module):
    
    def __init__(self,keys,embed_dim,n_classes=5):
        super().__init__()
        self.keys = keys
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.model = torch.load('vae_encoder.pkl')
        self.avg   = nn.AvgPool2d(kernel_size=7)
        self.embedding = nn.Embedding(self.n_classes, self.embed_dim)
    def forward(self,x):
        img     = x[self.keys[0]]
        cls_lab =  x[self.keys[1]][:,None]
        out     = self.model(img)
        out     = self.avg(out)
        emd_lab = self.embedding(cls_lab)
        
        out = out.squeeze(-1).squeeze(-1)
        # print(out.shape)
        # print(emd_lab.shape)
        fin_emb = torch.cat([out.unsqueeze(1),emd_lab],dim=1)
        
        fin_emb = fin_emb.view(out.shape[0],-1)
        
        return fin_emb.unsqueeze(1)

class Classifier_Embed_Guidance(nn.Module):

    def __init__(self,key):
        super().__init__()
        self.model = torch.load('cut_classifier.pkl')
        self.key = key
    def forward(self,x):
        x  = x[self.key]
        op = self.model(x)
        op = op.unsqueeze(1)
        
        return op


class Dual_Embedder_Pos(nn.Module):
    
    def __init__(self,keys,n_classes=5):
        super().__init__()
        self.keys = keys
        
        self.n_classes = n_classes
        self.model = torch.load('auto_kl_encoder.pkl').cuda()
        self.pos   = PositionalEncoding2D(5)
        
        nn.Conv2d(6, 1, 4, stride=2, padding=2)
#         self.avg   = nn.AvgPool2d(kernel_size=7)
#         self.embedding = nn.Embedding(self.n_classes, self.embed_dim)
    def forward(self,x):
        img     = x[self.keys[0]]
        cls_lab = x[self.keys[1]]
        out     = self.model(img)
        
        emd_lab = self.pos(torch.zeros(out.shape[1],out.shape[2],out.shape[3],self.n_classes))
#         print(emd_lab.shape)
        emd_lab = emd_lab.permute(-1,0,1,2)
#         print(emd_lab.shape)
        emd_lab = emd_lab[cls_lab,:,:,:].cuda()
        fin_out = out + emd_lab
        
        out_layer = nn.Conv2d(6, 1, 1, stride=2, padding=0).cuda()
        fin_out = out_layer(fin_out)
        # print("3333",fin_out.shape)
        fin_out = fin_out.view(fin_out.shape[0],fin_out.shape[1],fin_out.shape[2]*fin_out.shape[3])
        
        return fin_out




class Feats_Embedder(nn.Module):
    
    def __init__(self,key):
        super().__init__()
        self.key = key
        self.model = torch.load('vae_encoder.pkl')
        self.avg   = nn.AvgPool2d(kernel_size=7)
        
    def forward(self,x):
        img     = x[self.key]
        out     = self.model(img)
        out     = self.avg(out)
        out     = out.view(out.shape[0],-1)
        out     = out.unsqueeze(1)
        # print(out.shape)
        
        return out


class Dual_Embedder_SEP(nn.Module):
    
    def __init__(self,keys,embed_dim,n_classes=5):
        super().__init__()
        self.keys = keys
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.model = torch.load('vae_encoder.pkl')
        self.avg   = nn.AvgPool2d(kernel_size=7)
        self.embedding = nn.Embedding(self.n_classes, self.embed_dim)
    def forward(self,x):
        img     = x[self.keys[0]]
        cls_lab =  x[self.keys[1]][:,None]
        out     = self.model(img)
        out     = self.avg(out)
        emd_lab = self.embedding(cls_lab)
        out = out.view(out.shape[0],1,-1)
        emd_lab = emd_lab.view(emd_lab.shape[0],1,-1)
        fin_emb = torch.cat([out,emd_lab],dim=1)

        # out = out.squeeze(-1).squeeze(-1)
        # # print(out.shape)
        # # print(emd_lab.shape)
        # fin_emb = torch.cat([out.unsqueeze(1),emd_lab],dim=1)
        
        # fin_emb = fin_emb.view(out.shape[0],-1)
        
        return fin_emb


class Dual_Embedder_TIME(nn.Module):
    
    def __init__(self,keys):
        super().__init__()
        self.keys = keys
        
        
        self.model = torch.load('vae_encoder.pkl')
        self.avg   = nn.AvgPool2d(kernel_size=7)
        
    def forward(self,x):
        img     = x[self.keys[0]]
        cls_lab =  x[self.keys[1]][:,None]
        out     = self.model(img)
        out     = self.avg(out)
        
        out = out.view(out.shape[0],1,-1)
        
        return [out,cls_lab]