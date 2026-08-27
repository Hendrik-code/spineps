# High-Level API

The easiest way to use SPINEPS from Python: the one-call [`segment`][spineps.api.segment] function, a reusable
[`SpinepsPipeline`][spineps.api.SpinepsPipeline] (load the models once, segment many images), and grouped
configuration objects to toggle individual processing steps without long keyword-argument lists.

```python
import spineps

result = spineps.segment("sub-01_T2w.nii.gz")          # saves a derivatives folder next to the input
result = spineps.segment(nii, output_in_memory=True)   # or return the masks in memory
if result.success:
    semantic, vertebra = result.semantic, result.vertebra
```

## spineps.api

::: spineps.api

## spineps.config

::: spineps.config
