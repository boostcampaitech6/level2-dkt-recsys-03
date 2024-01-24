import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class ModelBase(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_tests = n_tests
        self.n_questions = n_questions
        self.n_tags = n_tags

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = hidden_dim, hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_test = nn.Embedding(n_tests + 1, intd)
        self.embedding_question = nn.Embedding(n_questions + 1, intd)
        self.embedding_tag = nn.Embedding(n_tags + 1, intd)

        # Concatentaed Embedding Projection
        self.comb_proj = nn.Linear(intd * 4, hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
    def forward(self, test, question, tag, correct, mask, interaction):
        batch_size = interaction.size(0)
        # Embedding
        embed_interaction = self.embedding_interaction(interaction.int())
        embed_test = self.embedding_test(test.int())
        embed_question = self.embedding_question(question.int())
        embed_tag = self.embedding_tag(tag.int())
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            dim=2,
        )
        X = self.comb_proj(embed)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 2,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        **kwargs
    ):
        super().__init__(
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.n_layers,
            num_attention_heads=self.n_heads,
            max_position_embeddings=max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

class Pos_encoding(nn.Module):
    def __init__(
            self,
            drop_out: float = 0.1,
            max_seq_len: float = 20,
            hidden_dim: int = 64
    ):
        super().__init__()
        self.dropout = drop_out
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.drop = nn.Dropout(drop_out)
        self.pos_emb = torch.zeros(self.max_seq_len, self.hidden_dim).unsqueeze(0) 
        #i/(10000^(2j/d)) <- j:col, i:row, d:dim 
        term = torch.arange(0, self.max_seq_len).unsqueeze(1)/torch.pow(10000, torch.arange(0,self.hidden_dim,2)/self.hidden_dim)
        self.pos_emb[:, :, 0::2] = torch.sin(term)
        self.pos_emb[:, :, 1::2] = torch.cos(term)

    def forward(self, input):
        out = input + self.pos_emb[:,:input.shape[1], :].to(input.device)
        return self.drop(out)

#add & norm layer for transformer  
class AddNorm(nn.Module):
    def __init__(
            self,
            drop_out: float = 0.1,
            norm_shape: int = 64
    ):
        super().__init__()
        self.drop_out = drop_out
        self.norm_shape = norm_shape

        self.dropout = nn.Dropout(self.drop_out)
        self.norm = nn.LayerNorm(self.norm_shape)
    
    def forward(self, original, res):
        return self.norm(self.dropout(res) + original)
    
#feed forward network
class FFN(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 64,
            out_dim: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, self.hidden_dim)
        )

    def forward(self, out, batch_size):
        out = out.view(-1, self.hidden_dim)
        out = self.ffn(out)
        out = out.view(batch_size,-1,self.hidden_dim)
        return out
    
#LQTR 구현
class LQTR(ModelBase):
    #encoding-> transformer(not positional), sequential-> LSTM 
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tests: int = 1538,
        n_questions: int = 9455,
        n_tags: int = 913,
        n_heads: int = 5,
        drop_out: float = 0.1,
        max_seq_len: float = 20,
        out_dim: float = 128,
        **kwargs
    ):
        super().__init__( 
            hidden_dim,
            n_layers,
            n_tests,
            n_questions,
            n_tags
        )
        self.n_heads = n_heads
        self.drop_out = drop_out
        self.max_seq_len = max_seq_len
        self.out_dim = out_dim

        #encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.pos_emb = Pos_encoding(self.drop_out, self.max_seq_len, self.hidden_dim)
        self.attn = nn.MultiheadAttention(self.hidden_dim, self.n_heads, dropout = self.drop_out,batch_first=True)
        self.addNorm1 = AddNorm(self.drop_out,self.hidden_dim)
        self.ffn = FFN(self.hidden_dim, self.out_dim)
        self.addNorm2 = AddNorm(self.drop_out,self.hidden_dim)
        #LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )
        #DNN
        self.dnn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim,1)
        )

    def forward(self, test, question, tag, correct, mask, interaction):
        X, batch_size = super().forward(test=test,
                                        question=question,
                                        tag=tag,
                                        correct=correct,
                                        mask=mask,
                                        interaction=interaction)
        
        self.q = self.query(X)
        self.k = self.key(X)
        self.v = self.value(X)
        #positional encoding
        #pos_encoded = self.pos_emb(X)

        #encoding(not positional emb) #use only last query
        attn_output,_ = self.attn(query=self.q[:,-1:,:],key=self.k,value=self.v)
        #add&norm
        addnorm1_output = self.addNorm1(X, attn_output)
        #FFN
        ffn_output = self.ffn(addnorm1_output, batch_size)
        #add&norm
        addnorm2_output = self.addNorm2(addnorm1_output, ffn_output)
        #lstm
        out, _ = self.lstm(addnorm2_output)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.dnn(out).view(batch_size, -1)

        return out
