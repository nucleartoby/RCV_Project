import pandas as pd
import numpy as np
import requests
from sec_edgar_downloader import Downloader

d1 = Downloader("FAR Technologies", "toby.manwaring02@gmail.com")

d1.get("8-K", "AAPL")