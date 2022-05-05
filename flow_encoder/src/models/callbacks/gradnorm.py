import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback


class GradNorm(Callback):
    """
    Callback adapting loss weights during the training for MTL.
    Implementation of https://arxiv.org/pdf/1711.02257.pdf.
    Note that: `pl_module` must have `loss_weights` and `task_loss` attributes
    and `model._get_shared_layer` method.

    Code adapted from: https://github.com/falkaer/artist-group-factors/
    """

    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def on_after_backward(self, trainer: Trainer, pl_module: LightningModule):
        """ """
        # Zero the w_i(t) gradients to update the weights using gradnorm loss
        pl_module.loss_weights.grad = 0.0 * pl_module.loss_weights.grad
        W = list(pl_module.model._get_shared_layer())

        norms = []
        for w_i, L_i in zip(pl_module.loss_weights, pl_module.task_loss):
            # Gradient of L_i(t) w.r.t. W
            gLgW = torch.autograd.grad(L_i, W, retain_graph=True)
            # G^{(i)}_W(t)
            norms.append(torch.norm(w_i * gLgW[0]))
        norms = torch.stack(norms)

        # Set L(0)
        if trainer.fit_loop.batch_idx == 0:
            self.initial_losses = pl_module.task_loss.detach()

        # Compute the constant term without accumulating gradients
        # as it should stay constant during back-propagation
        with torch.no_grad():
            # Loss ratios \curl{L}(t)
            loss_ratios = pl_module.task_loss / self.initial_losses

            # Inverse training rate r(t)
            inverse_train_rates = loss_ratios / loss_ratios.mean()

            constant_term = norms.mean() * (inverse_train_rates ** self.alpha)

        # Write out the gradnorm loss L_grad and set the weight gradients
        grad_norm_loss = (norms - constant_term).abs().sum()
        pl_module.loss_weights.grad = torch.autograd.grad(
            grad_norm_loss, pl_module.loss_weights
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
            normalize_coeff = (
                len(pl_module.loss_weights) / pl_module.loss_weights.sum()
            )
            pl_module.loss_weights.data = (
                pl_module.loss_weights.data * normalize_coeff
            )
