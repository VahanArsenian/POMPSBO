# POMPSBO

To run the project one shall firstly build the scmmab package. Run
```shell
python setup.py install
```
from scmmab directory.

After that, an experiment may be run by defining an instance of `experiments/experiment/POMPSExperiment`.
To reproduce the results please execute the corresponding file f rom `experiments` directory. For example:
```shell
python pomps_paper_example.py --no-smoke --n_iter 1200 --seed 42 
```

## Citations

```bibtex
@misc{contextualcausalbayesianoptimisation,
      title={Contextual Causal Bayesian Optimisation}, 
      author={Vahan Arsenyan and Antoine Grosnit and Haitham Bou-Ammar},
      year={2025},
      eprint={2301.12412},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2301.12412}, 
}
```
