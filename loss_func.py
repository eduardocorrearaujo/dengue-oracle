import torch 
import torch.nn as nn
from torch.distributions import LogNormal

def norm_cdf(x):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))

def crps_lognormal_torch(obs, mulog, sigmalog, eps=1e-6):
    # Garantir que todos os valores são positivos e estáveis numericamente
    obs = torch.clamp(obs, min=eps)
    sigmalog = torch.clamp(sigmalog, min=eps)

    ω = (torch.log(obs) - mulog) / sigmalog
    ex = 2.0 * torch.exp(mulog + 0.5 * sigmalog**2)

    cdf_ω = norm_cdf(ω)
    cdf_ω_minus_sigma = norm_cdf(ω - sigmalog)
    cdf_sigma_div_sqrt2 = norm_cdf(sigmalog / torch.sqrt(torch.tensor(2.0, device=sigmalog.device)))

    term1 = obs * (2.0 * cdf_ω - 1.0)
    term2 = ex * (cdf_ω_minus_sigma + cdf_sigma_div_sqrt2 - 1.0)

    return term1 - term2  # retorno é shape (batch, ...), não média ainda

class CRPSLogNormalLossNew(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, mu, sigma, target):
        crps = crps_lognormal_torch(target, mu, sigma, eps=self.eps)
        if self.reduction == 'mean':
            return crps.mean()
        elif self.reduction == 'sum':
            return crps.sum()
        else:
            return crps 
class LogNormalNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # to avoid log(0)

    def forward(self, mu, sigma, target):
        # Ensure stability: clamp sigma and target
        sigma = torch.clamp(sigma, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        log_target = torch.log(target)
        log_sigma = torch.log(sigma)

        nll = log_target + log_sigma + ((log_target - mu) ** 2) / (2 * sigma ** 2)
        return nll.mean()
                   
class LogNormalCRPSLoss(nn.Module):
    def __init__(self, samples = 100, eps = 1e-6):
        super().__init__()
        self.samples = samples
        self.eps = eps

    def forward(self, mu, sigma, target):
        # Clamp to ensure numerical stability
        sigma = torch.clamp(sigma, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        # Create LogNormal distribution
        dist = torch.distributions.LogNormal(mu, sigma)

        # Sample from the distribution: shape (samples, batch, predict_n)
        samples = dist.rsample((self.samples,))

        # Expand target to match samples shape: (1, batch, predict_n)
        target_expanded = target.unsqueeze(0)

        # First term: E|X - y|
        term1 = torch.mean(torch.abs(samples - target_expanded), dim=0)

        # Second term: 0.5 * E|X - X'|
        samples1 = samples.unsqueeze(1)  # (samples, 1, batch, predict_n)
        samples2 = samples.unsqueeze(0)  # (1, samples, batch, predict_n)
        abs_diffs = torch.abs(samples1 - samples2)  # (samples, samples, batch, predict_n)
        term2 = 0.5 * torch.mean(abs_diffs, dim=(0,1))

        # CRPS per example and per predicted step
        crps = term1 - term2  # shape: (batch, predict_n)

        # Return mean CRPS over batch and prediction horizon
        return crps.mean()

class WISLossFromDistribution(nn.Module):
    def __init__(self, alphas=[0.5, 0.2, 0.1, 0.05], reduction='mean', eps=1e-6):
        """
        alphas: list of significance levels, e.g., [0.5, 0.2, 0.1, 0.05] for 50%, 80%, 90%, 95% intervals
        reduction: 'mean', 'sum', or None
        eps: small value to ensure numerical stability
        """
        super().__init__()
        self.alphas = alphas
        self.reduction = reduction
        self.w0 = 0.5
        self.wks = [alpha / 2.0 for alpha in alphas]
        self.K = len(alphas)
        self.eps = eps

    def interval_score(self, lower, upper, target, alpha):
        width = upper - lower
        below = (target < lower).float()
        above = (target > upper).float()

        penalty = (2.0 / alpha) * (
            (lower - target) * below + (target - upper) * above
        )

        return width + penalty

    def forward(self, mu, sigma, target):
        # Clamp for numerical stability
        sigma = torch.clamp(sigma, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        dist = torch.distributions.LogNormal(mu, sigma)

        # Median: quantile 0.5
        q50 = dist.icdf(torch.tensor(0.5, device=mu.device))

        # Start WIS with median absolute error
        wis = self.w0 * torch.abs(q50 - target)

        for alpha, wk in zip(self.alphas, self.wks):
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2

            lower = dist.icdf(torch.tensor(lower_q, device=mu.device))
            upper = dist.icdf(torch.tensor(upper_q, device=mu.device))

            iscore = self.interval_score(lower, upper, target, alpha)
            wis = wis + wk * iscore

        wis = wis / (self.K + 0.5)

        if self.reduction == 'mean':
            return wis.mean()
        elif self.reduction == 'sum':
            return wis.sum()
        else:
            return wis

class IntervalScore(nn.Module):
    def __init__(self, alpha=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, lower, upper, target):
        # penalty terms
        below = (target < lower).float()
        above = (target > upper).float()

        penalty = (
            2.0 / self.alpha
            * ((lower - target) * below + (target - upper) * above)
        )

        score = (upper - lower) + penalty

        if self.reduction == 'mean':
            return score.mean()
        elif self.reduction == 'sum':
            return score.sum()
        else:
            return score

class MAEMedian(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, target):
        median = torch.exp(mu)  # median of LogNormal
        error = torch.abs(median - target)
        if self.reduction == 'mean':
            return error.mean()
        elif self.reduction == 'sum':
            return error.sum()
        else:
            return error


def get_prediction_interval(mu, sigma, alpha=0.1):
    dist = LogNormal(mu, sigma)
    lower = dist.icdf(torch.tensor(alpha / 2, device=mu.device))
    upper = dist.icdf(torch.tensor(1 - alpha / 2, device=mu.device))
    return lower, upper

class CombinedCRPSIntervalLoss(nn.Module):
    def __init__(self, alpha=0.1, crps_weight=1.0, interval_weight=1.0, eps=1e-6):
        super().__init__()
        self.crps_loss = LogNormalCRPSLoss(eps = eps)#CRPSLogNormalLossNew(eps=eps)
        self.interval_score = IntervalScore(alpha=alpha)
        self.alpha = alpha
        self.crps_weight = crps_weight
        self.interval_weight = interval_weight

    def forward(self, mu, sigma, target):
        crps = self.crps_loss(mu, sigma, target)

        lower, upper = get_prediction_interval(mu, sigma, self.alpha)
        interval = self.interval_score(lower, upper, target)

        return self.crps_weight * crps + self.interval_weight * interval

class MAEIntervalLoss(nn.Module):
    def __init__(self, alpha=0.1, mae_weight=1.0, interval_weight=1.0):
        super().__init__()
        self.mae = MAEMedian()
        self.interval = IntervalScore(alpha=alpha)
        self.alpha = alpha
        self.mae_weight = mae_weight
        self.interval_weight = interval_weight

    def forward(self, mu, sigma, target):
        # MAE from median
        mae_loss = self.mae(mu, target)

        # Interval score from quantiles
        lower, upper = get_prediction_interval(mu, sigma, self.alpha)
        interval_loss = self.interval(lower, upper, target)

        return self.mae_weight * mae_loss + self.interval_weight * interval_loss

class MAELoss(nn.Module):
    def __init__(self, alpha=0.1, mae_weight=1.0, interval_weight=1.0):
        super().__init__()
        self.mae = MAEMedian()

    def forward(self, mu, sigma, target):
        # MAE from median
        mae_loss = self.mae(mu, target)

        return mae_loss
