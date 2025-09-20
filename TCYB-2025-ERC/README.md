# Tracking Correlations between Multiple Data Streams through Evolutionary Regressor Chains

Source code of [Tracking Correlations between Multiple Data Streams through Evolutionary Regressor Chains](https://ieeexplore.ieee.org/document/11091452).

## Dependencies

The only dependency is [PyTorch](https://pytorch.org).

## Usage

The `EvolutionaryRegressorChains` class should be initialized with the following parameters.
- the number of chains
- the number of data streams
- a base learner

**Example**

```python
>>> from erc.utils import TrainStreams, BaseLearnerTrain
>>> from erc.model import EvolutionaryRegressorChains
>>> sydtrain = TrainStreams(device)
>>> erc_train = EvolutionaryRegressorChains(10, 8, BaseLearnerTrain)
>>> for i, data in enumerate(sydtrain):
>>>     print('  Processing sample {}...'.format(i + 1))
>>>     x, y = data
>>>     loss = erc_train.step(x, y)
```

## Citation

Please cite it as:
```bib
@ARTICLE{
  author={Zhang, Bin and Lu, Jie and Liu, Anjin and Yao, Xin and Zhang, Guangquan},
  journal={IEEE Transactions on Cybernetics}, 
  title={Tracking Correlations Between Multiple Data Streams Through Evolutionary Regressor Chains}, 
  year={2025},
  volume={55},
  number={9},
  pages={4078-4088},
  doi={10.1109/TCYB.2025.3587025}}
```
