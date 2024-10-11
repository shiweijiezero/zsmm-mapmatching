import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_clusters=32, hidden_dim=1024):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = Encoder()

        # Temporal-aware Network
        self.temporal_aware = TemporalAwareNetwork()

        # Spatial-aware Network
        self.spatial_aware = SpatialAwareNetwork()

        # Scenario-adaptive Experts
        self.scenario_experts = ScenarioAdaptiveExperts(num_clusters, hidden_dim)

        # Decoder
        self.decoder = Decoder()

    def forward(self, x_traj, x_road):
        # Encode inputs
        encoded = self.encoder(torch.cat([x_traj, x_road], dim=1))

        # Temporal-aware features
        h_s = self.temporal_aware(encoded, x_traj)

        # Spatial-aware features
        h_u = self.spatial_aware(encoded, x_road)

        # Scenario-adaptive experts
        z, kl_loss = self.scenario_experts(encoded)

        # Combine features
        combined_features = torch.cat([h_s, h_u, z], dim=1)

        # Decode
        calibrated_traj = self.decoder(combined_features, x_road)

        return calibrated_traj, kl_loss

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)

class TemporalAwareNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2
        )

    def forward(self, x, traj):
        x = self.cnn(x)
        # Apply RoPE (simplified version)
        b, c, h, w = x.shape
        positions = torch.arange(h*w, device=x.device).reshape(1, 1, h, w).repeat(b, c, 1, 1)
        x = x + positions
        x = x.flatten(2).permute(2, 0, 1)
        return self.transformer(x)

class SpatialAwareNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.mlp = nn.Linear(256, 2)  # Output mean and variance

    def forward(self, x, road):
        attn_output, _ = self.attention(x.flatten(2).permute(2, 0, 1),
                                        road.flatten(2).permute(2, 0, 1),
                                        road.flatten(2).permute(2, 0, 1))
        return self.mlp(attn_output.mean(dim=0))

class ScenarioAdaptiveExperts(nn.Module):
    def __init__(self, num_clusters, hidden_dim):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        self.cluster_mlp = nn.Linear(256, num_clusters)
        self.expert_mlp = nn.ModuleList([nn.Linear(256, hidden_dim*2) for _ in range(num_clusters)])

    def forward(self, x):
        w = F.softmax(self.cluster_mlp(x.mean(dim=(2, 3))), dim=1)

        z_params = torch.stack([expert(x.mean(dim=(2, 3))) for expert in self.expert_mlp])
        z_mu, z_logvar = z_params.chunk(2, dim=-1)

        z = self.reparameterize(z_mu, z_logvar)
        z = (w.unsqueeze(-1) * z).sum(dim=1)

        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return z, kl_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(1024 + 256, 1, kernel_size=3, padding=1)
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)

    def forward(self, x, road):
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, road.shape[2], road.shape[3])
        x = self.upsample(x)
        x = torch.cat([x, road], dim=1)

        # Cross-attention
        x_flat = x.flatten(2).permute(2, 0, 1)
        road_flat = road.flatten(2).permute(2, 0, 1)
        attn_output, _ = self.attention(x_flat, road_flat, road_flat)
        x = attn_output.permute(1, 2, 0).view_as(x)

        return torch.sigmoid(self.conv(x))