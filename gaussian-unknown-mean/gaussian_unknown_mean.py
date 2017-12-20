import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer as infer
import pyro.optim
import pyro.distributions as dist

import matplotlib.pyplot as plt


class Gaussian(nn.Module):
    def __init__(self, prior_mean, prior_std, observation_std):
        super(Gaussian, self).__init__()
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.observation_std = observation_std

        # set up layers for guide net
        self.fcn1 = nn.Linear(2, 10)
        self.fcn2 = nn.Linear(10, 20)
        self.fcn3 = nn.Linear(20, 10)
        self.fcn4 = nn.Linear(10, 5)
        self.fcn5 = nn.Linear(5, 2)

    def model(self, observation1=0, observation2=0):
        """
        makes a 1-D observation from a Gaussian prior under Gaussian noise

        - note that observation is fed in as keyword argument with same name
        - also note that the default value of observation cannot be None or it
          will see the observe as a sample
        """
        latent = pyro.sample("latent",
                             dist.normal,
                             self.prior_mean,
                             self.prior_std)

        observation1 = pyro.observe("observation1",
                                    dist.normal,
                                    obs=observation1,
                                    mu=latent,
                                    sigma=self.observation_std)

        observation2 = pyro.observe("observation2",
                                    dist.normal,
                                    obs=observation2,
                                    mu=latent,
                                    sigma=self.observation_std)
        return latent

    def forward(self, observation1=None, observation2=None):
        """
        guide for proposal distributions

        takes same arguments as the model
        """
        # observation should always be given a non-default values
        assert observation1 is not None
        assert observation2 is not None

        observation1 = observation1.view(1, 1)
        observation2 = observation2.view(1, 1)
        x = torch.cat((observation1, observation2), 1)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        x = F.relu(self.fcn4(x))
        x = self.fcn5(x)
        x = x.view(2)

        proposal_mean = x[0]

        log_proposal_std = x[1]
        proposal_std = log_proposal_std.exp()

        pyro.sample("latent",
                    dist.normal,
                    proposal_mean,
                    proposal_std)


gaussian = Gaussian(prior_mean=Variable(torch.Tensor([1])),
                    prior_std=Variable(torch.Tensor([5**0.5])),
                    observation_std=Variable(torch.Tensor([2**0.5])))

num_samples = 50  # number of samples to create empirical distribution

# do CSIS
csis = infer.CSIS(model=gaussian.model,
                  guide=gaussian,
                  num_samples=10)
csis.set_model_args()                       # the model has no arguments except the observes
csis.set_compiler_args(num_particles=10)
optim = torch.optim.Adam(gaussian.parameters(), lr=1e-3)    # optimiser that will be used in compilation
csis.compile(optim, num_steps=500)
csis_marginal = infer.Marginal(csis)                        # draws weighted traces using Pyro's built-in importance sampling
csis_samples = [csis_marginal(observation1=Variable(torch.Tensor([8])),
                              observation2=Variable(torch.Tensor([9]))).data[0] for _ in range(10000)]

# do Importance sampling:
is_posterior = infer.Importance(model=gaussian.model,
                                num_samples=num_samples)
is_marginal = infer.Marginal(is_posterior)
is_samples = [is_marginal(observation1=Variable(torch.Tensor([8])),
                          observation2=Variable(torch.Tensor([9]))).data[0] for _ in range(10000)]


plt.hist(csis_samples, range=(-10, 10), bins=100, color='r', normed=1, label="Inference Compilation")
plt.hist(is_samples, range=(-10, 10), bins=100, color='b', normed=1, label="Importance Sampling")
plt.legend()
plt.title("Gaussian Unknown Mean Predictions")
plt.savefig("plots/histogram.pdf")
