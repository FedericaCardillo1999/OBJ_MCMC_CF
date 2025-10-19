"""Centralize all the library imports"""
from __future__ import annotations
import sys
subj = f"sub-{sys.argv[1].zfill(2)}"  

import os
import io
import re
import time
import glob
import pickle
import cortex
import random
import itertools
import math as m
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize
from numba import jit
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from dataclasses import dataclass
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional
from nibabel.freesurfer.io import read_morph_data, write_morph_data
from CFandPRF import source_eccentricity, load_prf, filter_prf, PRFModel, adjusting_verticesandecc, benson_label_to_dict
from config import (MAIN_PATH, freesurfer_path, project, hemispheres, atlases, tasks, denoising_methods, target_visual_areas, runs, rois_list, ses, source_visual_area, load_one, max_eccentricity, benson_max, filter_source, cutoff_volumes, ncores, has_multiple_runs, subjects)
from OBJ_MCMC_CF.MCMC_CF import MCMC_CF_cluster