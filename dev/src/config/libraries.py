# === GLOBAL LIBRARIES IMPORTS

# Model imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import lightning as L
import torchvision
from torch import nn
import torchmetrics
import time
import os
from tqdm import tqdm
import cv2


# Fullstack UI imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
