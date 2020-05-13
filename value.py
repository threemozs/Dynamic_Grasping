import torch
import torch.nn as nn
import torch.nn.functional as F


class Value(nn.Module):
	def __init__(self, s_dim):
		super(Value, self).__init__()

		self.net = nn.Sequential(nn.Linear(s_dim, 5),
		                         nn.ReLU(),
		                         nn.Linear(5, 5),
		                         nn.ReLU(),
		                         nn.Linear(5, 1))

	def forward(self, s):
		"""

		:param s: [b, s_dim]
		:return:  [b, 1]
		"""
		s = s.double()
		value = self.net(s)

		return value
