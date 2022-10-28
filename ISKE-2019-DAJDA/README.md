# Drift Adaptation via Joint Distribution Alignment

Source code of [Drift Adaptation via Joint Distribution Alignment (DAJDA)](https://ieeexplore.ieee.org/document/9170335).

## Dependencies

Install the dependecies via
```python
pip install -r requirements.txt
```

## Usage

The class should be initialized with a classifier.
There are two methods, `fit` and `predict`.

**Example**

```python
>>> import numpy as np
>>> from sklearn.linear_model import LogisticRegression
>>> from dajda.model import DAJDA
>>> rng = np.random.default_rng(0)
>>> x = rng.normal(0, 1, (3, 3))
>>> y = rng.choice(2, 3)
>>> clf = DAJDA(LogisticRegression(random_state=0))
>>> clf.fit(x, y)
>>> x2 = rng.normal(0, 1, (10, 3))
>>> clf.predict(x2)
array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
```

## Citation

Please cite it as:
```bib
@INPROCEEDINGS{
  author={Zhang, Bin and Lu, Jie and Zhang, Guangquan},
  title={Drift Adaptation via Joint Distribution Alignment},
  booktitle={2019 IEEE 14th International Conference on Intelligent Systems and Knowledge Engineering (ISKE)},
  year={2019},
  pages={498-504},
  doi={10.1109/ISKE47853.2019.9170335}
}
```
