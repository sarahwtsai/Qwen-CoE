# Qwen-CoE

Name: Sarah Tsai \
Berkeley MIDS \
Course: DATASCI 266 Spring 2025

## Overview
Composition of Experts (CoE) models offer an efficient alternative to large monolithic language models by selecting specialized experts to generate responses, which minimizes inference costs. We present Qwen-CoE, a CoE architecture that uses a feedforward network (FFN) to route user inputs to one of three Qwen2.5 models. Here we evaluate the effectiveness of FFNs as an alternative to conventional two-step CoE routing mechanisms, and explore their ability to generalize beyond their training distribution, including when trained on partially synthetic data.