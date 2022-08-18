import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class GradNorm(Callback):
    """
    Callback adapting **sqrt** loss weights during the training for MTL.
    Implementation of https://arxiv.org/pdf/1711.02257.pdf.
    Note that: `pl_module` must have `loss_weights` and `task_loss` attributes
    and `model._get_shared_layer` method.

    Code adapted from: https://github.com/falkaer/artist-group-factors/
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        """ """
        # Set L(0) during the first iteration
        if trainer.fit_loop.batch_idx == 0:
            self.initial_losses = pl_module.task_loss.detach()

        # Zero the w_i(t) gradients to update the weights using gradnorm loss
        pl_module.sqrt_loss_weights.grad = (
            0.0 * pl_module.sqrt_loss_weights.grad
        )
        W = list(pl_module.model._get_shared_layer())

        # Compute the L2-norm of the gradient of the weighted single-task loss
        # with respect to the chosen weights
        norms = []
        for w_i, L_i in zip(pl_module.sqrt_loss_weights, pl_module.task_loss):
            # Gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
            # G^(i)_W(t)
            norms.append(torch.norm(w_i.square() * gLgW[0]))
        norms = torch.stack(norms)

        # Compute the constant term without accumulating gradients as it
        # should stay constant during back-propagation
        with torch.no_grad():
            # Loss ratios L(t)
            loss_ratios = pl_module.task_loss / self.initial_losses
            # Inverse training rates r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()
            # Residual term _G_W(t)*r_i(t)^alpha
            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        # Compute the gradnorm loss L_grad
        grad_norm_loss = (norms - constant_term).abs().sum()
        # Update weight gradients
        pl_module.sqrt_loss_weights.grad = torch.autograd.grad(
            grad_norm_loss, pl_module.sqrt_loss_weights
        )[0]

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx,
        unused=0,
    ):
        # Renormalize the gradient weights
        with torch.no_grad():
            square_normalize_coeff = (
                len(pl_module.loss_weights)
                / pl_module.sqrt_loss_weights.square().sum()
            )
            pl_module.sqrt_loss_weights.data = (
                pl_module.sqrt_loss_weights.data.square()
                * square_normalize_coeff
            ).sqrt()
