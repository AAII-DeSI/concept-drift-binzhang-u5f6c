# A Multi-stream Concept Drift Handling Framework via Data Sharing

Source code of [A Multi-stream Concept Drift Handling Framework via Data Sharing]().

## Dependencies

The only dependencies are [NumPy](https://numpy.org/),
[SciPy](https://scipy.org/) and [PyTorch](https://pytorch.org).

## Usage

The `StreamHandler` class should be initialized with a base learner.

**Example**

```python
>>> from random import choice
>>> from fuzzmdx.model import StreamHandler
>>> x_train, y_train = DataStreams[:train_size]
>>> hdlrs = [StreamHandler(linear_model.Ridge(alpha=1)) for _ in range(m)]
>>> for j, hdlr in enumerate(hdlrs):
>>>     jother = choice([jother for jother in range(m) if jother != j])
>>>     hdlr.fit(x_train[:, j, :], y_train[:, j], x_train[:, jother, :], y_train[:, jother])
>>> N = (n - train_size) // batch_size
>>> for i in range(N):
>>>     i1 = train_size + i * batch_size
>>>     i2 = i1 + batch_size
>>>     x, y = DataStreams[i1:i2]
>>>     drifted_streams = []
>>>     for j in range(m):
>>>         result, drift = hdlrs[j].score(x[:, j, :], y[:, j])
>>>         if drift:
>>>             drifted_streams.append(j)
>>>     for j in drifted_streams:
>>>         retrainset = [j for j in range(m) if j not in drifted_streams] + [j]
>>>         jother = choice([jother for jother in range(m) if jother != j])
>>>         x_retrain = x[:, retrainset, :].reshape(-1, d)
>>>         y_retrain = y[:, retrainset].reshape(-1)
>>>         wght = np.ones(len(retrainset) * batch_size)
>>>         for ki, k in enumerate(retrainset):
>>>             if k == j:
>>>                 break
>>>             k1 = ki * batch_size
>>>             k2 = k1 + batch_size
>>>             wght[k1:k2] *= hdlrs[k].score(x[:, j, :], y[:, j], True).mean()
>>>         hdlrs[j].fit(
>>>             x_retrain,
>>>             y_retrain,
>>>             x[:, jother, :],
>>>             y[:, jother],
>>>             sample_weight=wght
>>>         )
```

## Citation

Please cite it as:
```bib
Not published yet.
```
