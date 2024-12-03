#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

#DATASET INFO
DATASET = "fluxnet2015_daily"
folds=5

PROJECT = "tam_rl"

# FILES INFO
DATA_DIR = os.path.join("/","home", "kumarv", "vashi024", "tam_rl", "DATA")
#RAW_DIR = os.path.join("/", "home", "kumarv", "renga016", "Public", "DATA", "{}".format(DATASET), "RAW")
RAW_DIR = os.path.join("/", "home", "kumarv", "vashi024", "DATA")
# PREPROCESSED_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"PREPROCESSED")
RESULT_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(PROJECT),"MODEL")

PREPROCESSED_DIR = os.path.join("/", "home", "kumarv", "vashi024", "DATA","PREPROCESSED")
# RESULT_DIR = os.path.join("/", "home", "kumarv", "vashi024", "tam_rl","DATA", "fluxnet2015_daily", "tam_rl",  "RESULT")
# MODEL_DIR = os.path.join("/", "home", "kumarv", "vashi024", "DATA","MODEL")




if not os.path.exists(PREPROCESSED_DIR):
	os.makedirs(PREPROCESSED_DIR)
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

# TIME SERIES INFO
train_year = {"start":1989, "end":1999}
valid_year = {"start":1999, "end":2001}
test_year = {"start":2001, "end":2009}
window = 30
stride = 15
adapt_year = 1

# CHANNELS INFO
#channels_names = np.array(['P_ERA','LAI','VPD_ERA','TA_ERA','SW_IN_ERA','GPP_NT_VUT_REF'])
channels_names = np.array(['P_ERA','Lai','VPD_ERA','TA_ERA','SW_IN_ERA','GPP_NT_VUT_REF', "RECO_NT_VUT_REF", "pft_MF","pft_CRO","pft_CSH","pft_DBF","pft_EBF","pft_ENF","pft_GRA","pft_OSH","pft_SAV","pft_SNO","pft_WET","pft_WSA","climate_Arctic","climate_Continental","climate_Temperate","climate_Tropical","climate_Arid"])
channels = list(range(len(channels_names)))
dynamic_channels = channels[:5]
output_channels = [channels[5]]


no_normalize_channels = channels[5]
normalize_channels = [channels[:5] + channels[6:]] 

# LABELS INFO
add = 0.005
unknown = np.nan

# MODEL INFO
forward_code_dim = 350
latent_code_dim = 64
device = "cuda"
recon_weight = 0.1
contrastive_weight = 1.0
static_weight = 0.0
kl_weight = 1.0
forward_weight = 1.0		# KGSSL, VAE
dropout = 0.4

# TRAIN INFO
inits = 5
train = True
batch_size = 64
epochs = 100
learning_rate = 1e-3
meta_learning_rate = 1e-3


# INFERENCE INFO
runs = 100