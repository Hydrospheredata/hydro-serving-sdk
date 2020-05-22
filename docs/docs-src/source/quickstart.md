# Quickstart

```note:: If you haven't launched Hydrosphere.io platform, you can learn how to do it here - https://hydrosphere.io/serving-docs/latest/install/index.html. 
```

## Installation

Install the latest release of hydrosdk via pip
```
pip install hydrosdk
```
or you can install latest version from git by running
```
pip install git+git://github.com/Hydrospheredata/hydro-serving-sdk.git
```

## Using hydrosdk

To use hydrosdk, you must first import it connect to your Hydrosphere.io platform by 
creating a Cluster object.


```python
from hydrosdk.cluster import Cluster

# Provide your cluster address here
cluster = Cluster("my-cluster")
```

Now that you have established a connection to Hydrosphere platform via Cluster object, you can make manage your cluster.
 The following lists all model versions deployed to your platform and prints their information:
 
 ```python
from hydrosdk.modelversion import ModelVersion

# Print out model names and versions
for modelversion in ModelVersion.list_models(cluster=cluster):
    print(modelversion)
```

It's also easy to send data to your deployed models.
For example, the following loads a csv and sends all rows to your deployed model through predictor object,
 which is designed to make inference for your data.
 
```python
from hydrosdk.application import Application
import pandas as pd

app = Application.find_by_name(cluster, "my-model")
predictor = app.predictor()

df = pd.read_csv("path/to/data.csv")
for row in df.itertuples(index=False):
    predictor.predict(row._as_dict())
```

Other topics such as LocalModels, Metrics and ... will be covered in more detail in the following sections,
 so don't worry if you do not completely understand the examples.