#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARIES

# In[1]:


import numpy as np
import torch
import LSTM
from sklearn.metrics import pairwise_distances


# # LOSS FUNCTIONS

# ## CONTRASTIVE LOSS

# In[2]:


class SimCLRLoss(torch.nn.Module):
	def __init__(self, temperature):
		super(SimCLRLoss, self).__init__()
		self.temperature = temperature
		self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
		self.similarity = torch.nn.CosineSimilarity(dim=2)

	def mask_correlated_samples(self, batch_size):
		N = 2 * batch_size
		mask = torch.ones((N, N), dtype=bool)
		mask = mask.fill_diagonal_(0)

		for i in range(batch_size):
			mask[i, batch_size + i] = 0
			mask[batch_size + i, i] = 0
		return mask

	def forward(self, z):

		z = torch.nn.functional.normalize(z, p=2.0, dim=1)

		N = z.shape[0]
		batch_size = N//2

		sim = self.similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

		sim_i_j = torch.diag(sim, batch_size)
		sim_j_i = torch.diag(sim, -batch_size)

		positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
		mask = self.mask_correlated_samples(batch_size)
		negative_samples = sim[mask].reshape(N, -1)

		#SIMCLR
		labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long()

		logits = torch.cat((positive_samples, negative_samples), dim=1)
		loss = self.criterion(logits, labels)
		loss /= N

		return loss


# ## KL LOSS

# In[3]:


class KLLoss(torch.nn.Module):
	def __init__(self):
		super(KLLoss, self).__init__()

	def forward(self, z, mu, std):
		# 1. define the first two probabilities (in this case Normal for both)
		p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
		q = torch.distributions.Normal(mu, std)

		# 2. get the probabilities from the equation
		log_qzx = q.log_prob(z)
		log_pz = p.log_prob(z)

		# loss
		loss = (log_qzx - log_pz)
		loss = torch.mean(torch.sum(loss, dim=1), dim = 0)
		return loss


# # FORWARD MODELS

# ## LSTM MODELS

# <!-- ### LSTM -->

# In[4]:


class lstm(torch.nn.Module):

	def __init__(self, input_channels, hidden_dim, output_channels, dropout):
		super(lstm,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = torch.nn.LSTM(input_size=self.input_channels, hidden_size=self.hidden_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x_encoder, _ = self.encoder(x_dynamic)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# ### MYLSTM

# In[5]:


# class lstm(torch.nn.Module):

# 	def __init__(self, input_channels, hidden_dim, output_channels, dropout):
# 		super(lstm,self).__init__()

# 		# PARAMETERS
# 		self.input_channels = input_channels
# 		self.hidden_dim = hidden_dim
# 		self.output_channels = output_channels

# 		# LAYERS
# 		self.encoder = LSTM.LSTM(input_size=self.input_channels, hidden_size=self.hidden_dim, batch_first=True)
# 		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
# 		self.dropout = torch.nn.Dropout(p=dropout)

# 		# INITIALIZATION
# 		for m in self.modules():
# 			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
# 				torch.nn.init.xavier_uniform_(m.weight)

# 	def forward(self, x_dynamic):

# 		# GET SHAPES
# 		batch, window, _ = x_dynamic.shape

# 		# OPERATIONS
# 		x_encoder, _ = self.encoder(x_dynamic)
# 		x_encoder = self.dropout(x_encoder)
# 		out = self.out(x_encoder)
# 		out = out.view(batch, window, self.output_channels)

# 		return out


# ### MYCTLSTM

# In[6]:


# class ctlstm(torch.nn.Module):

# 	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
# 		super(ctlstm,self).__init__()

# 		# PARAMETERS
# 		self.input_dynamic_channels = input_dynamic_channels
# 		self.input_static_channels = input_static_channels
# 		self.hidden_dim = hidden_dim
# 		self.output_channels = output_channels

# 		# LAYERS
# 		self.encoder = LSTM.LSTM(input_size=self.input_dynamic_channels+self.input_static_channels, hidden_size=self.hidden_dim, batch_first=True)
# 		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
# 		self.dropout = torch.nn.Dropout(p=dropout)

# 		# INITIALIZATION
# 		for m in self.modules():
# 			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
# 				torch.nn.init.xavier_uniform_(m.weight)

# 	def forward(self, x_dynamic, x_static):

# 		# GET SHAPES
# 		batch, window, _ = x_dynamic.shape

# 		# OPERATIONS
# 		x = torch.cat((x_dynamic, x_static), dim=-1)
# 		x_encoder, _ = self.encoder(x)
# 		x_encoder = self.dropout(x_encoder)
# 		out = self.out(x_encoder)
# 		out = out.view(batch, window, self.output_channels)

# 		return out


# ### TAMLSTM

# In[7]:


class tamlstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
		super(tamlstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = LSTM.TAMLSTM(input_size=self.input_dynamic_channels, latent_size =self.input_static_channels,  hidden_size=self.hidden_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		# x = torch.cat((x_dynamic, x_static), dim=-1)
		x = x_dynamic
		x_encoder, _ = self.encoder(x,x_static)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# In[8]:


# class tamlstm(torch.nn.Module):

# 	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
# 		super(tamlstm,self).__init__()

# 		# PARAMETERS
# 		self.input_dynamic_channels = input_dynamic_channels
# 		self.input_static_channels = input_static_channels
# 		self.hidden_dim = hidden_dim
# 		self.output_channels = output_channels

# 		# LAYERS
# 		self.encoder = LSTM.TAMLSTM(input_size=self.input_dynamic_channels, latent_size =self.input_static_channels,  hidden_size=self.hidden_dim, batch_first=True)
# 		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
# 		self.dropout = torch.nn.Dropout(p=dropout)
        

# 		# INITIALIZATION
# 		for m in self.modules():
# 			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
# 				torch.nn.init.xavier_uniform_(m.weight)

# 	def forward(self, x_dynamic, x_static):

# 		# GET SHAPES
# 		batch, window, _ = x_dynamic.shape

# 		# OPERATIONS
# 		# x = torch.cat((x_dynamic, x_static), dim=-1)
# 		x = x_dynamic
# 		x_encoder, _ = self.encoder(x,x_static)
# 		x_encoder = self.dropout(x_encoder)
# 		out = self.out(x_encoder)
        
# 		out = out.view(batch, window, self.output_channels)

# 		return out


# In[ ]:


class tamlstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
		super(tamlstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = LSTM.TAMLSTM(input_size=self.input_dynamic_channels, latent_size =self.input_static_channels,  hidden_size=self.hidden_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)
		self.relu = torch.nn.ReLU()
		self.linear = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		# x = torch.cat((x_dynamic, x_static), dim=-1)
		x = x_dynamic
		x_encoder, _ = self.encoder(x,x_static)
		x_encoder = self.dropout(x_encoder)
		out = self.relu(self.linear(x_encoder))
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# ### CTLSTM

# In[9]:


class ctlstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
		super(ctlstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels

		# LAYERS
		self.encoder = torch.nn.LSTM(input_size=self.input_dynamic_channels+self.input_static_channels, hidden_size=self.hidden_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x = torch.cat((x_dynamic, x_static), dim=-1)
		x_encoder, _ = self.encoder(x)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# ### FILMLSTM

# In[10]:


class filmlstm(torch.nn.Module):

	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
		super(filmlstm,self).__init__()

		# PARAMETERS
		self.input_dynamic_channels = input_dynamic_channels
		self.input_static_channels = input_static_channels
		self.hidden_dim = hidden_dim
		self.output_channels = output_channels

		# LAYERS
		self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim)
		self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim)
		self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
		self.out = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
		self.dropout = torch.nn.Dropout(p=dropout)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Parameter):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x_dynamic, x_static):

		# GET SHAPES
		batch, window, _ = x_dynamic.shape

		# OPERATIONS
		x_dynamic_encoder = self.dynamic_encoder(x_dynamic)
		# x_dynamic_encoder = self.dropout(x_dynamic_encoder)
		x_static_encoder = self.static_encoder(x_static)
		# x_static_encoder = self.dropout(x_static_encoder)
		x_static_encoder = x_static_encoder + torch.ones_like(x_static_encoder)
		x = x_static_encoder*x_dynamic_encoder+x_static_encoder
		x_encoder, _ = self.encoder(x)
		x_encoder = self.dropout(x_encoder)
		out = self.out(x_encoder)
		out = out.view(batch, window, self.output_channels)

		return out


# In[11]:


# class filmlstm(torch.nn.Module):

# 	def __init__(self, input_dynamic_channels, input_static_channels, hidden_dim, output_channels, dropout):
# 		super(filmlstm,self).__init__()

# 		# PARAMETERS
# 		self.input_dynamic_channels = input_dynamic_channels
# 		self.input_static_channels = input_static_channels
# 		self.hidden_dim = hidden_dim
# 		self.output_channels = output_channels

# 		# LAYERS
# 		self.dynamic_encoder = torch.nn.Linear(in_features=self.input_dynamic_channels, out_features=self.hidden_dim)
# 		self.static_encoder = torch.nn.Linear(in_features=self.input_static_channels, out_features=self.hidden_dim)
# 		self.encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
# 		self.out = torch.nn.Linear(in_features=self.output_channels, out_features=self.output_channels)
# 		self.dropout = torch.nn.Dropout(p=dropout)
# 		self.relu = torch.nn.ReLU()
# 		self.linear = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_channels)
        

# 		# INITIALIZATION
# 		for m in self.modules():
# 			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Parameter):
# 				torch.nn.init.xavier_uniform_(m.weight)

# 	def forward(self, x_dynamic, x_static):

# 		# GET SHAPES
# 		batch, window, _ = x_dynamic.shape

# 		# OPERATIONS
# 		x_dynamic_encoder = self.dynamic_encoder(x_dynamic)
# 		# x_dynamic_encoder = self.dropout(x_dynamic_encoder)
# 		x_static_encoder = self.static_encoder(x_static)
# 		# x_static_encoder = self.dropout(x_static_encoder)
# 		x_static_encoder = x_static_encoder + torch.ones_like(x_static_encoder)
# 		x = x_static_encoder*x_dynamic_encoder+x_static_encoder
# 		x_encoder, _ = self.encoder(x)
# 		x_encoder = self.dropout(x_encoder)
# 		out = self.relu(self.linear(x_encoder))
# 		out = self.out(x_encoder)
# 		out = out.view(batch, window, self.output_channels)

# 		return out


# # INVERSE MODELS

# ## AE

# 

# In[12]:


class ae(torch.nn.Module):
	def __init__(self, input_channels, hidden_dim, code_dim, output_channels, device):
		super(ae,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.code_dim = code_dim
		self.output_channels = output_channels
		self.device = device

		# LAYERS
		self.instance_encoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=self.input_channels, out_features=self.hidden_dim),
			torch.nn.BatchNorm1d(self.hidden_dim),
			torch.nn.LeakyReLU(0.2)
		)
		self.temporal_encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True)	# AE
		self.code_linear = torch.nn.Linear(self.hidden_dim, self.code_dim)																		# AE
		self.decode_linear = torch.nn.Linear(self.code_dim, self.hidden_dim)																	# AE
		self.temporal_decoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)						# AE
		self.instance_decoder = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.input_channels)									# AE
		self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		# GET SHAPES
		batch, window, _ = x.shape

		# OPERATIONS

		x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)		# ENCODE
		_, x_encoder = self.temporal_encoder(x_encoder)													# ENCODE
		enc_vec = torch.sum(x_encoder[0], dim=0)														# ENCODE

		z = self.code_linear(enc_vec)																	# CODE_VEC

		static_out = self.static_out(z)																	# STATIC DECODE

		decode_vec = self.decode_linear(z)																# DECODE
		out = torch.zeros(batch, window, self.input_channels).to(self.device)							# DECODE
		input = torch.unsqueeze(torch.zeros_like(decode_vec), dim=1)									# DECODE
		h = (torch.unsqueeze(decode_vec, dim=0), torch.unsqueeze(torch.zeros_like(decode_vec), dim=0))	# DECODE
		for step in range(window):																		# DECODE
			input, h = self.temporal_decoder(input, h)													# DECODE
			out[:,step] = self.instance_decoder(input.squeeze())										# DECODE

		return z, enc_vec, static_out, out


# ## AE_SIMCLR

# In[13]:


class ae_simclr(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, code_dim, output_channels, device):
        super(ae_simclr,self).__init__()

        # PARAMETERS
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        self.output_channels = output_channels
        self.device = device

        # LAYERS
        self.instance_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_channels, out_features=self.hidden_dim),
            torch.nn.BatchNorm1d(self.hidden_dim),
            torch.nn.LeakyReLU(0.2)
        )
        self.temporal_encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True)	# AE
        self.code_linear = torch.nn.Linear(self.hidden_dim, self.code_dim)																		# AE
        self.decode_linear = torch.nn.Linear(self.code_dim, self.hidden_dim)																	# AE
        self.temporal_decoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)						# AE
        self.instance_decoder = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.input_channels)									# AE
        self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)
        self.sim_layers = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=self.code_dim, out_features=self.code_dim),
        )

        # INITIALIZATION
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):

        # GET SHAPES
        batch, window, _ = x.shape

        # OPERATIONS

        x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)		# ENCODE
        _, x_encoder = self.temporal_encoder(x_encoder)													# ENCODE
        enc_vec = torch.sum(x_encoder[0], dim=0)														# ENCODE

        z = self.code_linear(enc_vec)																	# CODE_VEC
        sim_vec = self.sim_layers(z)

        static_out = self.static_out(z)																	# STATIC DECODE

        decode_vec = self.decode_linear(z)																# DECODE
        out = torch.zeros(batch, window, self.input_channels).to(self.device)							# DECODE
        input = torch.unsqueeze(torch.zeros_like(decode_vec), dim=1)									# DECODE
        h = (torch.unsqueeze(decode_vec, dim=0), torch.unsqueeze(torch.zeros_like(decode_vec), dim=0))	# DECODE
        for step in range(window):																		# DECODE
            input, h = self.temporal_decoder(input, h)													# DECODE
            out[:,step] = self.instance_decoder(input.squeeze())										# DECODE

        return z, enc_vec,sim_vec, static_out, out


# ## VAE

# In[14]:


class vae(torch.nn.Module):
	def __init__(self, input_channels, hidden_dim, code_dim, output_channels, device):
		super(vae,self).__init__()

		# PARAMETERS
		self.input_channels = input_channels
		self.hidden_dim = hidden_dim
		self.code_dim = code_dim
		self.output_channels = output_channels
		self.device = device

		# LAYERS
		self.instance_encoder = torch.nn.Sequential(
			torch.nn.Linear(in_features=self.input_channels, out_features=self.hidden_dim),
			torch.nn.BatchNorm1d(self.hidden_dim),
			torch.nn.LeakyReLU(0.2)
		)
		self.temporal_encoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True)	# VAE
		self.mu = torch.nn.Linear(self.hidden_dim, self.code_dim)																				# VAE
		self.log_var = torch.nn.Linear(self.hidden_dim, self.code_dim)																			# VAE
		self.decode_linear = torch.nn.Linear(self.code_dim, self.hidden_dim)																	# VAE
		self.temporal_decoder = torch.nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=True)						# VAE
		self.instance_decoder = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.input_channels)									# VAE
		self.static_out = torch.nn.Linear(in_features=self.code_dim, out_features=self.output_channels)

		# INITIALIZATION
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
				torch.nn.init.xavier_uniform_(m.weight)

	def forward(self, x):

		# GET SHAPES
		batch, window, _ = x.shape

		# OPERATIONS

		x_encoder = self.instance_encoder(x.view(-1, self.input_channels)).view(batch, window, -1)		# ENCODE
		_, x_encoder = self.temporal_encoder(x_encoder)													# ENCODE
		enc_vec = torch.sum(x_encoder[0], dim=0)														# ENCODE

		mu, log_var = self.mu(enc_vec), self.log_var(enc_vec)											# SAMPLE Z
		std = torch.exp(log_var/2)																		# SAMPLE Z
		z = mu + std * torch.randn_like(std)															# SAMPLE Z

		static_out = self.static_out(z)																	# STATIC DECODE

		decode_vec = self.decode_linear(z)																# DECODE
		out = torch.zeros(batch, window, self.input_channels).to(self.device)							# DECODE
		input = torch.unsqueeze(torch.zeros_like(decode_vec), dim=1)									# DECODE
		h = (torch.unsqueeze(decode_vec, dim=0), torch.unsqueeze(torch.zeros_like(decode_vec), dim=0))	# DECODE
		for step in range(window):																		# DECODE
			input, h = self.temporal_decoder(input, h)													# DECODE
			out[:,step] = self.instance_decoder(input.squeeze())										# DECODE

		return z, mu, std, static_out, out


# # TEST MODELS

# In[15]:


if __name__ == "__main__":
	batch = 10
	window = 365
	channels = list(range(30))
	static_channels = channels[:20]
	dynamic_channels = channels[20:29]
	output_channels = [channels[-1]]
	data = torch.randn(batch, window, len(static_channels)+len(dynamic_channels)+len(output_channels))
	print(data.shape, "DATA")

	forward_code_dim = 128
	code_dim = 32
	device = torch.device("cpu")
	dropout = 0.4

	architecture = "lstm"
	model = globals()[architecture](input_channels=len(dynamic_channels), hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(x_dynamic=data[:,:,dynamic_channels].to(device))
	print(data[:, :, dynamic_channels].shape, out.shape, "#:{}".format(pytorch_total_params), architecture)
    
	architecture = "ctlstm"
	model = globals()[architecture](input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(x_dynamic=data[:,:,dynamic_channels].to(device), x_static=data[:,:,static_channels].to(device))
	print(data.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)
    
	architecture = "tamlstm"
	model = globals()[architecture](input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(x_dynamic=data[:,:,dynamic_channels].to(device), x_static=data[:,:,static_channels].to(device))
	print(data.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "filmlstm"
	model = globals()[architecture](input_dynamic_channels=len(dynamic_channels), input_static_channels=len(static_channels), hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	out = model(x_dynamic=data[:,:,dynamic_channels].to(device), x_static=data[:,:,static_channels].to(device))
	print(data.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "ae"
	model = globals()[architecture](input_channels=len(dynamic_channels)+len(output_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	code_vec, enc_vec, static_out, out = model(x=data[:, :, dynamic_channels+output_channels].to(device))
	print(data.shape, code_vec.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "ae_simclr"
	model = globals()[architecture](input_channels=len(dynamic_channels)+len(output_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	code_vec, enc_vec, sim_vec, static_out, out = model(x=data[:, :, dynamic_channels+output_channels].to(device))
	print(data.shape, code_vec.shape, sim_vec.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "vae"
	model = globals()[architecture](input_channels=len(dynamic_channels)+len(output_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	model = model.to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	code_vec, mu, std, static_out, out = model(x=data[:, :, dynamic_channels+output_channels].to(device))
	print(data.shape, code_vec.shape, mu.shape, std.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "kgssl_ctlstm"
	inverse_model = globals()["ae"](input_channels=len(dynamic_channels)+len(output_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	inverse_model = inverse_model.to(device)
	pytorch_total_params = sum(p.numel() for p in inverse_model.parameters() if p.requires_grad)
	forward_model = globals()["ctlstm"](input_dynamic_channels=len(dynamic_channels), input_static_channels=code_dim, hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	forward_model = forward_model.to(device)
	pytorch_total_params += sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
	code_vec, _, static_out, out = inverse_model(x=data[:, :, dynamic_channels+output_channels].to(device))
	out = forward_model(x_dynamic=data[:, :, dynamic_channels].to(device), x_static=code_vec[:data.shape[0]].unsqueeze(1).repeat(1, window, 1).to(device))
	print(data.shape, code_vec.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)

	architecture = "vae_ctlstm"
	inverse_model = globals()["vae"](input_channels=len(dynamic_channels)+len(output_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	inverse_model = inverse_model.to(device)
	pytorch_total_params = sum(p.numel() for p in inverse_model.parameters() if p.requires_grad)
	forward_model = globals()["ctlstm"](input_dynamic_channels=len(dynamic_channels), input_static_channels=code_dim, hidden_dim=forward_code_dim, output_channels=len(output_channels), dropout=dropout)
	forward_model = forward_model.to(device)
	pytorch_total_params += sum(p.numel() for p in forward_model.parameters() if p.requires_grad)
	code_vec, mu, std, static_out, out = inverse_model(x=data[:, :, dynamic_channels+output_channels].to(device))
	out = forward_model(x_dynamic=data[:, :, dynamic_channels].to(device), x_static=code_vec[:data.shape[0]].unsqueeze(1).repeat(1, window, 1).to(device))
	print(data.shape, code_vec.shape, static_out.shape, out.shape, "#:{}".format(pytorch_total_params), architecture)


# # COPY MODEL PARAMETERS

# In[ ]:


if __name__ == "__main__":

	architecture = "ae"
	model1 = globals()[architecture](input_channels=len(dynamic_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	model1 = model1.to(device)

	architecture = "vae"
	model2 = globals()[architecture](input_channels=len(dynamic_channels), hidden_dim=forward_code_dim, code_dim=code_dim, output_channels=len(static_channels), device=device)
	model2 = model2.to(device)

	model2.load_state_dict(model1.state_dict(), strict=False)

