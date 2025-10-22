import torch
import torch.nn as nn
from torch.autograd import Variable


class deeplog(nn.Module):
    def __init__(self, num_templates, embedding_dim, hidden_size, num_layers, num_keys,
                 num_metrics=29, metric_dim=64, nhead=4, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_metrics = num_metrics
        self.metric_dim = metric_dim

        # ===== Embedding =====
        self.template_emb = nn.Embedding(num_templates, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.seq_proj = nn.Linear(hidden_size, metric_dim)

        # ===== Metrics =====
        self.metric_proj = nn.Linear(1, metric_dim)
        self.metric_type_emb = nn.Embedding(num_metrics, metric_dim)  

        # ===== Cross Attention =====
        self.cross_seq2met = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)
        self.cross_met2seq = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)

        self.seq_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )
        self.met_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 2 * metric_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_keys)
        )

    def forward(self, features, device):
        seq_ids = features[0].squeeze(-1).long()  # [B, T]
        seq_emb = self.template_emb(seq_ids)  # [B, T, embedding_dim]

        B, T, _ = seq_emb.size()
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        lstm_out, _ = self.lstm(seq_emb, (h0, c0))  # [B, T, H]
        lstm_last = lstm_out[:, -1, :]  # [B, H]
        seq_proj = self.seq_proj(lstm_out)  # [B, T, D]

        metrics = features[1].float()  # [B, M]
        metrics_unsq = metrics.unsqueeze(-1)  # [B, M, 1]
        metric_tokens = self.metric_proj(metrics_unsq)  # [B, M, D]

        pos = torch.arange(0, self.num_metrics, device=device).unsqueeze(0).expand(B, -1)
        metric_tokens = metric_tokens + self.metric_type_emb(pos)

        # ---- Cross Attention ----
        # Metrics -> Sequence
        q1 = metric_tokens.transpose(0,1)
        k1v1 = seq_proj.transpose(0,1)
        attn_m2s, _ = self.cross_seq2met(q1, k1v1, k1v1)
        attn_m2s = attn_m2s.transpose(0,1)  # [B, M, D]

        # Sequence -> Metrics
        q2 = seq_proj.transpose(0,1)
        k2v2 = metric_tokens.transpose(0,1)
        attn_s2m, _ = self.cross_met2seq(q2, k2v2, k2v2)
        attn_s2m = attn_s2m.transpose(0,1)  # [B, T, D]

        met_summary = self.met_agg(attn_m2s.mean(dim=1))  # [B, D]
        seq_summary_last_step = attn_s2m[:, -1, :] # [B, D]
        seq_summary = self.seq_agg(seq_summary_last_step)

        combined = torch.cat((lstm_last, met_summary, seq_summary), dim=1)  
        out = self.fc(combined)  
        return out
    
class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys,
                 num_metrics=29, metric_dim=64, nhead=4, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_metrics = num_metrics

        self.template_emb = nn.Embedding(num_keys, metric_dim)

        self.lstm0 = nn.LSTM(metric_dim, hidden_size, num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.metric_proj = nn.Linear(1, metric_dim)
        self.metric_type_emb = nn.Embedding(num_metrics, metric_dim)

        self.cross_seq2met = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)
        self.cross_met2seq = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)

        self.seq_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )
        self.met_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size + 2 * metric_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_keys)
        )

    def forward(self, features, device):
        input0, input1, metrics = features[0], features[1], features[2]
        input0_emb = self.template_emb(input0.squeeze(-1).long())  

        def run_lstm(x, lstm_layer):
            B = x.size(0)
            h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
            out, _ = lstm_layer(x, (h0, c0))
            return out, out[:, -1, :]

        out0, last0 = run_lstm(input0_emb, self.lstm0)
        out1, last1 = run_lstm(input1, self.lstm1)

        seq_combined = torch.cat((out0, out1), dim=1)  
        seq_proj = seq_combined  

        B = metrics.size(0)
        metrics_unsq = metrics.unsqueeze(-1)
        metric_tokens = self.metric_proj(metrics_unsq)
        pos = torch.arange(0, self.num_metrics, device=device).unsqueeze(0).expand(B, -1)
        metric_tokens = metric_tokens + self.metric_type_emb(pos)

        q1 = metric_tokens.transpose(0, 1)
        k1v1 = seq_proj.transpose(0, 1)
        attn_m2s, _ = self.cross_seq2met(q1, k1v1, k1v1)
        attn_m2s = attn_m2s.transpose(0, 1)

        q2 = seq_proj.transpose(0, 1)
        k2v2 = metric_tokens.transpose(0, 1)
        attn_s2m, _ = self.cross_met2seq(q2, k2v2, k2v2)
        attn_s2m = attn_s2m.transpose(0, 1)

        met_summary = self.met_agg(attn_m2s.mean(dim=1))
        seq_summary = self.seq_agg(attn_s2m[:, -1, :])

        combined = torch.cat((last0, last1, met_summary, seq_summary), dim=1)
        out = self.fc(combined)
        return out

class robustlog(nn.Module):
    def __init__(self, num_templates, embedding_dim, hidden_size, num_layers, num_keys,
                 num_metrics=29, metric_dim=64, nhead=4, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_metrics = num_metrics
        self.metric_dim = metric_dim

        self.template_emb = nn.Embedding(num_templates, embedding_dim)

        self.lstm0 = nn.LSTM(46, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.seq_proj = nn.Linear(hidden_size, metric_dim)

        self.metric_proj = nn.Linear(1, metric_dim)
        self.metric_type_emb = nn.Embedding(num_metrics, metric_dim)  

        self.cross_seq2met = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)
        self.cross_met2seq = nn.MultiheadAttention(metric_dim, nhead, dropout=dropout, batch_first=False)

        self.seq_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )
        self.met_agg = nn.Sequential(
            nn.Linear(metric_dim, metric_dim),
            nn.ReLU(),
            nn.LayerNorm(metric_dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 2 * metric_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_keys)
        )

    def forward(self, features, device):
        # ---- Sequentials ----
        input0 = features[0] 

        B, T, _ = input0.size()
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=device)
        lstm_out, _ = self.lstm0(input0, (h0, c0)) 
        lstm_last = lstm_out[:, -1, :]  
        seq_proj = self.seq_proj(lstm_out)  

        # ---- Metrics ----
        metrics = features[1].float() 
        metrics_unsq = metrics.unsqueeze(-1)  
        metric_tokens = self.metric_proj(metrics_unsq)  

        pos = torch.arange(0, self.num_metrics, device=device).unsqueeze(0).expand(B, -1)
        metric_tokens = metric_tokens + self.metric_type_emb(pos)

        q1 = metric_tokens.transpose(0,1)
        k1v1 = seq_proj.transpose(0,1)
        attn_m2s, _ = self.cross_seq2met(q1, k1v1, k1v1)
        attn_m2s = attn_m2s.transpose(0,1)  

        q2 = seq_proj.transpose(0,1)
        k2v2 = metric_tokens.transpose(0,1)
        attn_s2m, _ = self.cross_met2seq(q2, k2v2, k2v2)
        attn_s2m = attn_s2m.transpose(0,1)  

        met_summary = self.met_agg(attn_m2s.mean(dim=1)) 
        seq_summary_last_step = attn_s2m[:, -1, :] 
        seq_summary = self.seq_agg(seq_summary_last_step)

        combined = torch.cat((lstm_last, met_summary, seq_summary), dim=1)  
        out = self.fc(combined)  
        return out