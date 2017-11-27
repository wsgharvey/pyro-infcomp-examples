import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pyro
import pyro.infer as infer
import pyro.optim
import pyro.distributions as dist

import matplotlib
import matplotlib.pyplot as plt

from PIL import Image

import numpy as np


class Gaussian(nn.Module):
    def __init__(self, prior_mean, prior_var, observation_var):
        super(Gaussian, self).__init__()
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.observation_var = observation_var

        # set up layers for guide net
        self.fcn1 = nn.Linear(1, 10)
        self.fcn2 = nn.Linear(10, 20)
        self.fcn3 = nn.Linear(20, 10)
        self.fcn4 = nn.Linear(10, 5)
        self.fcn5 = nn.Linear(5, 2)

    def model(self, observation=Variable(torch.Tensor([0]))):
        """
        makes a 1-D observation from a Gaussian prior under Gaussian noise

        - note that observation is fed in as keyword argument with same name
        - also note that the default value of observation cannot be None or it
          will see the observe as a sample
        """
        latent = pyro.sample("latent",
                             dist.normal,
                             self.prior_mean,
                             self.prior_var)

        observation = pyro.observe("observation",
                                   dist.normal,
                                   obs=observation,
                                   mu=latent,
                                   sigma=self.observation_var)

    def forward(self, observation=None):
        """
        guide for proposal distributions

        takes same arguments as the model
        """
        # observation should always be given a non-default values
        assert observation is not None

        x = observation.view(-1, 1)
        x = F.relu(self.fcn1(x))
        x = F.relu(self.fcn2(x))
        x = F.relu(self.fcn3(x))
        x = F.relu(self.fcn4(x))
        x = self.fcn5(x)
        x = x.view(2)

        proposal_mean = x[0]

        log_proposal_var = x[1]
        proposal_var = log_proposal_var.exp()

        pyro.sample("latent",
                    dist.normal,
                    proposal_mean,
                    proposal_var)


gaussian = Gaussian(prior_mean=Variable(torch.Tensor([0])),
                    prior_var=Variable(torch.Tensor([1])),
                    observation_var=Variable(torch.Tensor([0.1])))

csis = infer.CSIS(model=gaussian.model,
                  guide=gaussian,
                  optim=torch.optim.Adam)

csis.compile(num_steps=1000,
             num_particles=10)
