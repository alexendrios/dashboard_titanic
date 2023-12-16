# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:18:08 2023

@author: Alexandre
"""

import pandas as pd
import streamlit as st

arquivo = '../data/titanic.csv'
df = pd.read_csv(arquivo, sep=',')

print(df)