# ContextsPOI: Contextual Information on POI Recommendation


![ContextsPOI](https://github.com/rahmanidashti/ContextImpact/blob/master/images/banner.png)

## How to run the framework?

**All of the codes to run the models and processing are located in `codes` folder.**

1. At first step, we need to run the contextual models to get the preference scores of users on POIs based on each contextual influence. To do this, we provide all the contextual models in the `contextsModels` package which are in `lib`. When you run `saveScores.py`, all contextual models which are located in `lib` compute the users' preference score, then they will be saved as `NumPy arrays (.npy)` in `/model_combiner/contexts/dataset` and the dataset can be `Yelp` or `Gowalla`, indicates on which dataset your run the models and get these results. The size of these produces `NumPy arrays` is equal to `user_num * poi_num`.
2. The second step is the embedding of users and POIs that we need them as for the embedding layer of user and POIs in Neural Network based approaches. To embed the user and location into the latent feature space, you can run the `embedding.py` from the `MFEmbedder` package. the output is two Numpy arrays `U.npy` and `L.npy` which are saved in `/embeddings/dataset/`. 
3. Run the base models (MF and NN) to achive results on linear and non-linear models. To do this, you can run `main.py` in `codes` and save the results automatically to `Numpy arrays`. All of this process will be done automatically with code, for example, for `Neural Network` based models the code first loads  the `embedding` vectors of users and items.
4. Combine all possible combination of MF and NN models with contextual models separately. In order to combine differenc contextual models with each other and also linear and non-linear models we can use the code on the `model_cobiner` package. What we need to do is to config the model to evaluate models and context. In `model_combiner.py` we can define our specific configuration, we can select the dataset, base models, contextual models. Then, the results will be written under the `results` folder.

In the next version of our codes, we are going to make it easier to run, using the command-line arguments.

## Prerequisites

You will need below libraries to be installed before running the application:

- Python >= 3.8
- Numpy >= 1.17.3
- Tensorflow >= 2.2.0
- SciPy >= 1.4.1

For a simple solution, you can simply run the below command in the root directory:

```python
pip install -r prerequisites.txt
```

## Citation
If you find **ContextsPOI** useful for your research or development, please cite the following [paper](https://arxiv.org/):

```
@inproceedings{rahmani2021ContextsPOI,
  title={A Systematic Analysis on the Impact of Contextual Information on Point-of-Interest Recommendation},
  author={Hossein A. Rahmani, Mohammad Aliannejadi, Mitra Baratchi, Fabio Crestani},
  booktitle={TBA},
  year={2021}
}
```

<!-- ## TODOs
- [X] The release of base models source codes
- [X] Add context models to save the scores
- [X] final version code release after the acceptance of the paper
- [ ] Dataset characteristic will be added into the dataset readme file  
- [ ] User behaviour analysis part will be available after the acceptance of the paper.
- [ ] Add command-line run for the context models -->


## Team
* Hossein A. Rahmani, Web Intelligence Group, University College London, United Kingdom (h.rahmnai@ucl.ac.uk)
* Mohammad Aliannejadi, IRLab, University of Amsterdam, The Netherlands (m.aliannejadi@uva.nl)
* Mitra Baratchi, ADA Research Group, Leiden University, The Netherlands (m.baratchi@liacs.leidenuniv.nl)
* Fabio Crestani, IR-USI, Università della Svizzera italiana (USI), Switzerland (fabio.crestani@usi.ch)

## Acknowledgements
This work was in part supported by the Swiss State Secretariat for Education, Research and Innovation (SERI) mobility grant between Switzerland and Iran, and in part by the NWO (No. 016.Vidi.189.039 and No. 314-99-301).
