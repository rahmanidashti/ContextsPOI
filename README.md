# ContextsPOI: Contextual Information on POI Recommendation


![ContextsPOI](https://github.com/rahmanidashti/ContextImpact/blob/master/images/banner.png)

## How to run the framework?
1. The first step is to run the contextual models to get the preference scores of users on POIs based on each contextual infoluence. To do this, we provide all the contextual models in the `XXX` packages which is in `XXX`. When you run `XX`, all contextual models which are located in `XXX` compute the users' preference score then they will be saved as `numpy` arrays in `XXX`. The size of these produces numpy arrays is equal to `user_num` * `poi_num`. 
2. XX
3. XXX
4. XXXX

## User and Item Latent Feature Vectors
run `embedding.py` in `MFEmbeder`

## Prerequisites

You will need below libraries to be installed before running the application:

- Python >= 3.4
- NumPy >= 1.19
- SciPy >= 1.6
- PyInquirer >= 1.0.3

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

## TODOs
- [X] The release of base models source codes
- [X] Add context models to save the scores
- [ ] Add command-line run for the context models 
- [ ] final version code release after the acceptance of the paper
- [ ] User behaviour analysis part will be available after the acceptance of the paper.


## Team
* Hossein A. Rahmani, Web Intelligence Group, University College London, United Kingdom (h.rahmnai@ucl.ac.uk)
* Mohammad Aliannejadi, IRLab, University of Amsterdam, The Netherlands (m.aliannejadi@uva.nl)
* Mitra Baratchi, ADA Research Group, Leiden University, The Netherlands (m.baratchi@liacs.leidenuniv.nl)
* Fabio Crestani, IR-USI, Università della Svizzera italiana (USI), Switzerland (fabio.crestani@usi.ch)

## Acknowledgements
TBA
