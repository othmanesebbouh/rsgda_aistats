# RSGDA - Randomized Stochastic Gradient Descent Ascent

Code for the Randomized Gradient Descent Ascent, AISTATS 2022 (Othmane Sebbouh, Marco Cuturi, Marco Cuturi).
Based on haven-ai by Issam Laradji.

It is possible to either run the experiments from scratch, or see the figures in the ``precomputed_figures`` folder.

## Running the experiments from scratch

- For the experiments generating figure 1 and figure 2, run the following commands:
```
python trainval.py -e fig1_proba -d data -sb fig1
python trainval.py -e fig1_loop -d data -sb fig1
python trainval.py -e fig2 -d data -sb fig2
```

- For the experiments generating figures 3 and 4, please follow the following link (https://drive.google.com/drive/folders/10g7b7kR7krfRIggxWNmOsYNBFO9gMgN0?usp=sharing), and tun the included Colab Notebook ```training.ipynb```. We use a Colab notebook because we were unable to run pykeops on our server.
The data for these experiments can be downloaded from the following link: https://tpreports.nexus.ethz.ch/download/scim/data/tupro/

## Generating the plots
For each experiment, go to the corresponding folder, and run the jupyter notebook ```generate_figure_#```
