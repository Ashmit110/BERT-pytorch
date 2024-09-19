import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from classifier_head import ClassifierHead


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.classifier=ClassifierHead(self.hidden) # the second dimention should be that of the word embedding

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        
        
        '''-----------------ashmit entered code--------------------------------------'''    
        # Assuming `bert_output` is the output from the BERT model (shape: [B, T, E])
        # You can extract the [CLS] token embedding as follows:
        cls_token_embedding = x[:, 0, :]  # Shape becomes [B, E]
        

        # Forward pass through the classifier head
        output = self.classifier(cls_token_embedding)

        # Output shape will be [B, 1], representing the probability for each sample
        '''ashmit left the code'''
        return x