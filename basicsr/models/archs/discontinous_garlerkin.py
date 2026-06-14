"""Discontinuous Galerkin (DG) window operators with interface numerical flux."""
import torch
import torch.nn as nn

from basicsr.models.archs.galerkin import LayerNorm, window_partition, window_reverse


class DG_Win_SRNO(nn.Module):
    """
    DG-Window SRNO
    Window-local SRNO + DG flux between windows (operator-level)
    """
    def __init__(self, dim, heads=4, window=8, init_tau=0.25, flux_mode="jump", learnable_tau=True, boundary_type="neumann", init_kappa=1.0, flux_sign="neg"):
        super().__init__()
        assert dim % heads == 0
        assert flux_sign in ("neg", "pos"), "flux_sign must be 'neg' (legacy) or 'pos' (standard SIP convention)"

        self.boundary_type = boundary_type

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.scale = window * window
        self.flux_mode=flux_mode
        # Convention switch for the SIP penalty term in `interface_flux`.
        #   'neg' (default, legacy): F = -tau*(KR-KL)  — older runs / unroll3 (37.72) live here
        #   'pos' (standard SIP):    F = +tau*(KR-KL)  — matches DG textbooks, init=+0.25 starts
        #                                                 in the coercive regime
        # Only changes the *parameterization* of tau (sign), not the operator at convergence.
        self.flux_sign = flux_sign
        self._flux_sign_val = -1.0 if flux_sign == "neg" else +1.0

        self.qkv_proj = nn.Conv2d(dim, 3 * dim, 1)

        # DG coefficient (learnable)
        if flux_mode in ["jump", "avg_jump", "perona_malik"]:
            if learnable_tau:
                self.tau = nn.Parameter(torch.tensor(init_tau))
            else:
                self.tau = init_tau

        # Perona-Malik edge threshold (operator-level)
        if flux_mode == "perona_malik":
            if learnable_tau:
                self.kappa = nn.Parameter(torch.tensor(float(init_kappa)))
            else:
                self.kappa = init_kappa

        self.out_proj1 = nn.Conv2d(dim, dim, 1)
        self.out_proj2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        self.kln = LayerNorm((self.heads, 1, self.head_dim))
        self.vln = LayerNorm((self.heads, 1, self.head_dim))

    def interface_flux(self, KL, KR):
        """
        KL, KR: [..., heads, d, d]
        return F: same shape

        SIP-DG penalty term sign is governed by self._flux_sign_val:
            'neg' (legacy):     F = -tau * (KR - KL)
            'pos' (standard):   F = +tau * (KR - KL)
        Operator at convergence is identical up to tau ↔ -tau reparameterization.
        """
        if self.flux_mode == "central":
            F = 0.5 * (KL + KR)

        elif self.flux_mode == "jump":
            F = self._flux_sign_val * self.tau * (KR - KL)

        elif self.flux_mode == "avg_jump":
            F = 0.5 * (KL + KR) + self._flux_sign_val * self.tau * (KR - KL)

        elif self.flux_mode == "perona_malik":
            # Operator-level Perona-Malik: edge-preserving diffusion on SRNO operator
            #   F = sign · τ · g(‖K_R - K_L‖²) · (K_R - K_L)
            #   g(s) = 1 / (1 + s/κ²) — large operator-jump → low diffusion (preserve discontinuity)
            diff = KR - KL                                              # [..., heads, d, d]
            grad_norm2 = (diff ** 2).mean(dim=(-1, -2), keepdim=True)   # per-head Frobenius²
            kappa_val = self.kappa if torch.is_tensor(self.kappa) else torch.tensor(self.kappa, device=diff.device)
            g = 1.0 / (1.0 + grad_norm2 / (kappa_val ** 2 + 1e-8))
            F = self._flux_sign_val * self.tau * g * diff

        elif self.flux_mode == "upwind":
            # direction weight (P0, operator-level)
            alpha = torch.sigmoid(
                KR.mean(dim=(-1, -2)) - KL.mean(dim=(-1, -2))
            )                       # [..., heads]
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            F = alpha * KR + (1.0 - alpha) * KL

        else:
            raise ValueError(f"Unknown flux_mode {self.flux_mode}")

        return F

    def boundary_flux(self, K_inner, side):
        """
        K_inner: [..., heads, d, d]
        side: "left" | "right" | "up" | "down"
        """
        if self.boundary_type == "neumann":
            # zero normal flux
            return torch.zeros_like(K_inner)

        elif self.boundary_type == "dirichlet":
            # ghost state = 0
            K_ghost = torch.zeros_like(K_inner)
            return self.interface_flux(K_inner, K_ghost)

        elif self.boundary_type == "periodic":
            raise RuntimeError("Periodic BC handled separately")

        else:
            raise ValueError(self.boundary_type)

    def p0_dg_flux(self, K):
        """
        K: [B, Gh, Gw, heads, d, d]
        return flux: same shape
        """
        flux = torch.zeros_like(K)

        # =========================
        # x-interfaces (i,j)|(i,j+1)
        # =========================
        KL = K[:, :, :-1]
        KR = K[:, :,  1:]

        F = self.interface_flux(KL, KR)

        flux[:, :, :-1] += F
        flux[:, :,  1:] -= F

        # =========================
        # y-interfaces (i,j)|(i+1,j)
        # =========================
        KU = K[:, :-1, :]
        KD = K[:,  1:, :]

        F = self.interface_flux(KU, KD)

        flux[:, :-1, :] += F
        flux[:,  1:, :] -= F

        if self.boundary_type == "periodic":
            # x-direction wrap
            F = self.interface_flux(K[:, :, -1], K[:, :, 0])
            flux[:, :, -1] += F
            flux[:, :,  0] -= F

            # y-direction wrap
            F = self.interface_flux(K[:, -1, :], K[:, 0, :])
            flux[:, -1, :] += F
            flux[:,  0, :] -= F
        else:
            # left / right boundaries
            flux[:, :, 0]  += self.boundary_flux(K[:, :, 0],  side="left")
            flux[:, :, -1] += self.boundary_flux(K[:, :, -1], side="right")

            # up / down boundaries
            flux[:, 0, :]  += self.boundary_flux(K[:, 0, :],  side="up")
            flux[:, -1, :] += self.boundary_flux(K[:, -1, :], side="down")

        return flux

    def forward(self, x, name='0'):
        B, C, H, W = x.shape
        bias = x

        # ---- QKV ----
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1)
        q, k, v = qkv.chunk(3, dim=-1)

        # ---- window partition ----
        q = window_partition(q, self.window)\
                .reshape(-1, self.window*self.window, self.heads, self.head_dim)\
                .permute(0, 2, 1, 3)

        k = window_partition(k, self.window)\
                .reshape(-1, self.window*self.window, self.heads, self.head_dim)\
                .permute(0, 2, 1, 3)

        v = window_partition(v, self.window)\
                .reshape(-1, self.window*self.window, self.heads, self.head_dim)\
                .permute(0, 2, 1, 3)
        
        k = self.kln(k)
        v = self.vln(v)

        # ---- element-local operator ----
        # K_e: [Bw, heads, d, d]
        K = torch.matmul(k.transpose(-2, -1), v) / self.scale

        # ---- DG flux on operator grid ----
        Gh, Gw = H // self.window, W // self.window
        K = K.view(B, Gh, Gw, self.heads, self.head_dim, self.head_dim)

        K = K + self.p0_dg_flux(K)

        K = K.view(-1, self.heads, self.head_dim, self.head_dim)

        # ---- apply operator ----
        out = torch.matmul(q, K)

        # ---- reverse windows ----
        out = out.permute(0, 2, 1, 3)\
                 .reshape(-1, self.window, self.window, C)
        out = window_reverse(out, self.window, H, W)

        # ---- output ----
        out = out.permute(0, 3, 1, 2) + bias
        out = self.out_proj2(self.act(self.out_proj1(out))) + bias
        return out


class DG_Win_SRNO_FaceWise(nn.Module):
    """
    Face-wise DG-Window SRNO (Operator-level DG)
    Flux modes:
        - central  : central (average) flux
        - jump     : SIP jump-only flux (default)
        - avg_jump : SIP avg + jump flux
    """

    def __init__(self, dim, heads=16, window=8,
                 init_tau=0.25,
                 flux_mode="jump", 
                 learnable_tau=True,
                 boundary_type="neumann"):
        super().__init__()
        assert dim % heads == 0
        assert flux_mode in ["central", "jump", "avg_jump", "upwind"]

        self.boundary_type = boundary_type

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.scale = window * window
        self.flux_mode = flux_mode

        self.qkv_proj = nn.Conv2d(dim, 3 * dim, 1)

        # DG penalty coefficient
        if flux_mode in ["jump", "avg_jump"]:
            if learnable_tau:
                self.tau = nn.Parameter(torch.tensor(init_tau))
            else:
                self.tau = init_tau

        # output projection
        self.out_proj1 = nn.Conv2d(dim, dim, 1)
        self.out_proj2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()

        # per-head LN
        self.kln = LayerNorm((self.heads, 1, self.head_dim))
        self.vln = LayerNorm((self.heads, 1, self.head_dim))

        # face indices
        faces = self._build_face_indices(window)
        self.register_buffer("face_idx_left", faces["left"])
        self.register_buffer("face_idx_right", faces["right"])
        self.register_buffer("face_idx_top", faces["top"])
        self.register_buffer("face_idx_bottom", faces["bottom"])

    # -------------------------------------------------------
    # helpers
    # -------------------------------------------------------
    @staticmethod
    def _build_face_indices(p: int):
        idx = torch.arange(p * p).reshape(p, p)
        return {
            "left": idx[:, 0].reshape(-1),
            "right": idx[:, -1].reshape(-1),
            "top": idx[0, :].reshape(-1),
            "bottom": idx[-1, :].reshape(-1),
        }

    @staticmethod
    def _face_operator(k, v, face_idx):
        k_f = k[:, :, face_idx, :]
        v_f = v[:, :, face_idx, :]
        p = face_idx.numel()
        return torch.matmul(k_f.transpose(-2, -1), v_f) / float(p)

    @staticmethod
    def _reshape_to_grid(K, B, Gh, Gw):
        return K.view(B, Gh, Gw, K.shape[1], K.shape[2], K.shape[3])

    @staticmethod
    def _reshape_from_grid(Kg):
        B, Gh, Gw, h, d1, d2 = Kg.shape
        return Kg.view(B * Gh * Gw, h, d1, d2)

    def boundary_flux(self, K_inner, side):
        """
        K_inner: [..., heads, d, d]
        side: "left" | "right" | "up" | "down"
        """
        if self.boundary_type == "neumann":
            # zero normal flux
            return torch.zeros_like(K_inner)

        elif self.boundary_type == "dirichlet":
            # ghost state = 0
            K_ghost = torch.zeros_like(K_inner)
            return self.interface_flux(K_inner, K_ghost)

        elif self.boundary_type == "periodic":
            raise RuntimeError("Periodic BC handled separately")

        else:
            raise ValueError(self.boundary_type)

    def interface_flux(self, KL, KR):
        """
        KL, KR: [..., heads, d, d]
        return F: same shape
        """
        if self.flux_mode == "central":
            F = 0.5 * (KL + KR)

        elif self.flux_mode == "jump":
            F = - self.tau * (KR - KL)

        elif self.flux_mode == "avg_jump":
            F = 0.5 * (KL + KR) - self.tau * (KR - KL)

        elif self.flux_mode == "perona_malik":
            # Operator-level Perona-Malik: edge-preserving diffusion on SRNO operator
            #   F = -τ · g(‖K_R - K_L‖²) · (K_R - K_L)
            #   g(s) = 1 / (1 + s/κ²) — large operator-jump → low diffusion (preserve discontinuity)
            diff = KR - KL                                              # [..., heads, d, d]
            grad_norm2 = (diff ** 2).mean(dim=(-1, -2), keepdim=True)   # per-head Frobenius²
            kappa_val = self.kappa if torch.is_tensor(self.kappa) else torch.tensor(self.kappa, device=diff.device)
            g = 1.0 / (1.0 + grad_norm2 / (kappa_val ** 2 + 1e-8))
            F = -self.tau * g * diff

        elif self.flux_mode == "upwind":
            # direction weight (P0, operator-level)
            alpha = torch.sigmoid(
                KR.mean(dim=(-1, -2)) - KL.mean(dim=(-1, -2))
            )                       # [..., heads]
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            F = alpha * KR + (1.0 - alpha) * KL

        else:
            raise ValueError(f"Unknown flux_mode {self.flux_mode}")

        return F

    # -------------------------------------------------------
    # face-wise flux (core)
    # -------------------------------------------------------
    def _compute_face_flux(self, K_Lg, K_Rg, K_Tg, K_Bg):
        """
        K_*g: [B, Gh, Gw, heads, d, d]
        return: [B, Gh, Gw, heads, d, d]
        """
        flux = torch.zeros_like(K_Lg)

        # =========================
        # horizontal interfaces: (i,j)|(i,j+1)
        # =========================
        KL = K_Rg[:, :, :-1]    # left cell's RIGHT face
        KR = K_Lg[:, :,  1:]    # right cell's LEFT face

        F = self.interface_flux(KL, KR)

        flux[:, :, :-1] += F
        flux[:, :,  1:] -= F

        # =========================
        # vertical interfaces: (i,j)|(i+1,j)
        # =========================
        KU = K_Bg[:, :-1, :]    # upper cell's BOTTOM face
        KD = K_Tg[:,  1:, :]    # lower cell's TOP face

        F = self.interface_flux(KU, KD)

        flux[:, :-1, :] += F
        flux[:,  1:, :] -= F

        if self.boundary_type == "periodic":
            # x-direction wrap
            F = self.interface_flux(KR[:, :, -1], KL[:, :, 0])
            flux[:, :, -1] += F
            flux[:, :,  0] -= F

            # y-direction wrap
            F = self.interface_flux(KD[:, -1, :], KU[:, 0, :])
            flux[:, -1, :] += F
            flux[:,  0, :] -= F
        else:
            # left / right boundaries
            flux[:, :, 0]  += self.boundary_flux(KL[:, :, 0],  side="left")
            flux[:, :, -1] += self.boundary_flux(KR[:, :, -1], side="right")

            # up / down boundaries
            flux[:, 0, :]  += self.boundary_flux(KU[:, 0, :],  side="up")
            flux[:, -1, :] += self.boundary_flux(KD[:, -1, :], side="down")

        return flux

    def forward(self, x, name="0"):

        B, C, H, W = x.shape
        bias = x
        p = self.window
        assert H % p == 0 and W % p == 0

        # ---- QKV ----
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1)
        q, k, v = qkv.chunk(3, dim=-1)

        # ---- window partition ----
        q = window_partition(q, p).reshape(-1, p * p, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = window_partition(k, p).reshape(-1, p * p, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = window_partition(v, p).reshape(-1, p * p, self.heads, self.head_dim).permute(0, 2, 1, 3)

        k = self.kln(k)
        v = self.vln(v)

        # ---- volume operator ----
        K_vol = torch.matmul(k.transpose(-2, -1), v) / float(p * p)

        # ---- face operators ----
        K_L = self._face_operator(k, v, self.face_idx_left)
        K_R = self._face_operator(k, v, self.face_idx_right)
        K_T = self._face_operator(k, v, self.face_idx_top)
        K_B = self._face_operator(k, v, self.face_idx_bottom)

        Gh, Gw = H // p, W // p
        K_Lg = self._reshape_to_grid(K_L, B, Gh, Gw)
        K_Rg = self._reshape_to_grid(K_R, B, Gh, Gw)
        K_Tg = self._reshape_to_grid(K_T, B, Gh, Gw)
        K_Bg = self._reshape_to_grid(K_B, B, Gh, Gw)

        # ---- DG flux ----
        K_flux_g = self._compute_face_flux(K_Lg, K_Rg, K_Tg, K_Bg)
        K_flux   = self._reshape_from_grid(K_flux_g)

        # ---- final operator ----
        K_final = K_vol + K_flux

        # ---- apply operator ----
        out = torch.matmul(q, K_final)

        # ---- reverse windows ----
        out = out.permute(0, 2, 1, 3).reshape(-1, p, p, C)
        out = window_reverse(out, p, H, W)

        # ---- output ----
        out = out.permute(0, 3, 1, 2) + bias
        out = self.out_proj2(self.act(self.out_proj1(out))) + bias
        return out
