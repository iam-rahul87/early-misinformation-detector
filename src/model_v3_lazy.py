"""
MisinfoNet V3 — Gap-Aware Lazy Routing

Routing decision uses BOTH coverage AND gap:
  - coverage >= tau  → KB has seen something similar
  - |gap| >= gap_tau → KB can clearly distinguish real from fake

Only when BOTH conditions are true does the claim go to RAG.
If coverage is high but gap is near zero (KB is confused), it
falls through to the neural path instead.

Also fixes: softmax temperature bug at inference (was dividing
by self.T which made model artificially overconfident).
"""

import numpy as np
from src.nn import MLP, sigmoid, softmax, cross_entropy, cross_entropy_grad


class MisinfoNetV3Lazy:
    def __init__(self, rng, rag_module, cnn_extractor, rnn_extractor, rvnn_extractor,
                 neural_in_dim=93, coverage_threshold=0.15, gap_threshold=0.20,
                 lambda_rag=0.5, dropout=0.2, temp_start=5.0, temp_end=0.05):

        self.tau      = coverage_threshold
        self.gap_tau  = gap_threshold       # NEW — minimum |gap| to trust RAG
        self.lam      = lambda_rag
        self.T        = temp_start
        self.T_start  = temp_start
        self.T_end    = temp_end

        # Store extractors so predict() can call them selectively
        self.rag_mod  = rag_module
        self.cnn      = cnn_extractor
        self.rnn      = rnn_extractor
        self.rvnn     = rvnn_extractor

        self.rag_head    = MLP([6, 32, 16, 2],               rng, dropout)
        self.neural_head = MLP([neural_in_dim, 256, 128, 2],  rng, dropout)

        self._t        = 0
        self._training = True

    def train(self):
        self._training = True
        self.rag_head.train(); self.neural_head.train()

    def eval(self):
        self._training = False
        self.rag_head.eval(); self.neural_head.eval()

    def anneal_temperature(self, epoch, total_epochs):
        frac   = epoch / max(total_epochs - 1, 1)
        self.T = self.T_start * (self.T_end / self.T_start) ** frac

    # ── used during training (features pre-computed for speed) ────────────
    def forward_precomputed(self, rag_feat, cnn_feat, rnn_feat, rvnn_feat, coverage):
        rag_logits = self.rag_head.forward(rag_feat)
        neural_in  = np.hstack([cnn_feat, rnn_feat, rvnn_feat, rag_feat])
        neu_logits = self.neural_head.forward(neural_in)

        if self._training:
            alpha  = sigmoid((coverage - self.tau) / self.T)[:, None]
            final  = alpha * rag_logits + (1 - alpha) * neu_logits
        else:
            mask   = (coverage >= self.tau)[:, None].astype(np.float32)
            final  = mask * rag_logits + (1 - mask) * neu_logits
            alpha  = (coverage >= self.tau).astype(np.float32)

        return dict(logits=final, rag_logits=rag_logits, neu_logits=neu_logits,
                    neural_in=neural_in, rag_feat=rag_feat,
                    coverage=coverage, alpha=alpha)

    def loss_and_step(self, rag_feat, cnn_feat, rnn_feat, rvnn_feat,
                      coverage, labels, lr):
        self._t += 1
        out = self.forward_precomputed(rag_feat, cnn_feat, rnn_feat, rvnn_feat, coverage)

        L_main  = cross_entropy(out["logits"], labels)
        covered = coverage >= self.tau
        L_rag   = cross_entropy(out["rag_logits"][covered], labels[covered]) \
                  if covered.sum() > 1 else 0.0

        alpha = out["alpha"]
        if isinstance(alpha, np.ndarray) and alpha.ndim == 1:
            alpha = alpha[:, None]

        dL    = cross_entropy_grad(out["logits"], labels)
        d_rag = alpha * dL
        d_neu = (1 - alpha) * dL

        if covered.sum() > 1:
            d_aux          = np.zeros_like(out["rag_logits"])
            d_aux[covered] = self.lam * cross_entropy_grad(
                out["rag_logits"][covered], labels[covered])
            d_rag = d_rag + d_aux

        self.rag_head.backward(d_rag);    self.rag_head.step(lr, self._t)
        self.neural_head.backward(d_neu); self.neural_head.step(lr, self._t)

        preds   = out["logits"].argmax(1)
        rag_acc = (out["rag_logits"][covered].argmax(1) == labels[covered]).mean() \
                  if covered.sum() > 0 else float("nan")

        return dict(loss=L_main + self.lam * L_rag, acc=(preds==labels).mean(),
                    rag_acc=rag_acc, coverage=covered.mean())

    # ── GAP-AWARE LAZY INFERENCE ──────────────────────────────────────────
    def predict(self, texts: list[str]) -> dict:
        """
        Step 1: RAG features for ALL texts (always, cheap)
        Step 2: Dual routing condition —
                RAG path  if coverage >= tau  AND  |gap| >= gap_tau
                Neural    otherwise (low coverage OR ambiguous gap)
        Step 3: RAG head  → verdict for RAG-routed claims
        Step 4: CNN+RNN+RVNN + neural head → verdict for neural-routed claims

        The gap check prevents the RAG head from making decisions when the
        KB matches both real and fake entries equally (gap ~ 0), which was
        the primary source of false positives in the original model.
        """
        self.eval()
        N = len(texts)

        # Step 1 — RAG features (always, cheap)
        rag_feat, coverage = self.rag_mod.extract_features(texts)  # (N,6), (N,)

        # Step 2 — get raw gap for each claim (index 5 = real_max - fake_max)
        # must use raw (pre-StandardScaler) gap to stay on the 0-1 similarity scale
        raw_gaps = np.array([
            self.rag_mod._raw_features(t)[5]
            for t in texts
        ], dtype=np.float32)

        # Dual routing condition:
        #   coverage >= tau   → KB has seen something similar
        #   |gap| >= gap_tau  → KB clearly favours one side over the other
        # Both must be true — high coverage with gap~0 means KB is confused → Neural
        rag_mask = (coverage >= self.tau) & (np.abs(raw_gaps) >= self.gap_tau)
        neu_mask = ~rag_mask
        n_rag    = int(rag_mask.sum())
        n_neu    = int(neu_mask.sum())

        final = np.zeros((N, 2), dtype=np.float32)

        # Step 3 — RAG head (always computed, only used when rag_mask=True)
        rag_logits      = self.rag_head.forward(rag_feat)
        final[rag_mask] = rag_logits[rag_mask]

        # Step 4 — Neural path for everything else
        if n_neu > 0:
            neu_texts = [texts[i] for i in np.where(neu_mask)[0]]

            cnn_feat  = self.cnn.transform(neu_texts)
            rnn_feat  = self.rnn.transform(neu_texts)
            rvnn_feat = self.rvnn.transform(neu_texts)

            neu_in          = np.hstack([cnn_feat, rnn_feat, rvnn_feat,
                                         rag_feat[neu_mask]])
            final[neu_mask] = self.neural_head.forward(neu_in)

        # BUG FIX: removed /self.T — dividing by 0.05 inflated logits x20
        # and made the model artificially overconfident at inference
        probs = softmax(final/self.T)

        return dict(
            predictions    = probs.argmax(1),
            probabilities  = probs,
            routed_to_rag  = rag_mask,
            rag_coverage   = float(rag_mask.mean()),
            n_rag_routed   = n_rag,
            n_neu_routed   = n_neu,
            raw_gap        = raw_gaps,   # per-claim gap for inspection
        )

    def param_count(self):
        return self.rag_head.param_count() + self.neural_head.param_count()
