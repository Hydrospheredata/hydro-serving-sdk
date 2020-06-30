# Sample tutorial
This tutorial will show you how to use hydrosdk. You will learn how to upload your code as a model to Hydrosphere.io,
 setup your client code to make inference, and attach a monitoring metric to your model.
 
 ``` important:: This tutorials was written for hydrosdk==2.3.2
```
```note:: If you haven't launched Hydrosphere.io platform, please do so before proceeding with this tutorial.
 You can learn how to do by checking documentation here - https://hydrosphere.io/serving-docs/latest/install/index.html. 
```

 ## Upload your code as a Local Model
 
First we need to connect to the Hydrosphere.io platform by creating [Cluster](hydrosdk/hydrosdk.cluster) object.

 ```python
from hydrosdk.cluster import Cluster
from grpc import ssl_channel_credentials

# This cluster instance uses both - HTTP and GRPC APIs. Latter is used only for sending data to a deployed model
cluster = Cluster("http-cluster-address",
                  grpc_address="grpc-cluster-address", ssl=True,
                  grpc_credentials=ssl_channel_credentials())
```

Next we need to write the script which will be uploaded to the Hydrosphere platform.
For the simplicity of this tutorial let's imagine that our model will be calculating
 square root of a double value provided.

We'll call this script 'func_main.py' and put it in "model/src/func_main.py".

``` code-block:: python
     :linenos:
     :caption: model/src/func_main.py

     from math import sqrt
        
     def infer(x):
        return {"y": sqrt(x)}
```


Our file structure at this point should look like this:

```
.
└── model
    └── src
        └── func_main.py
```

To let hydrosdk know which files needs to be uploaded to the Hydrosphere.io platform we will capture all 
necessary file paths in a `payload`. We can also specify a common prefix to all paths in a `payload` in a `path` argument.
```python
path = "model/"
payload = ['src/func_main.py']
```

Hydrosphere Serving has a strictly typed inference engine, so before uploading our model we need to specify it's signature with
`SignatureBuilder`. Signature contains information about which method inside your `func_main.py` should be called,
 as well as what its inputs and outputs shapes and types are.
 
```python
from hydrosdk.contract import SignatureBuilder, ModelContract

signature = SignatureBuilder('infer') \
                .with_input('x', 'double', "scalar") \
                .with_output('y', 'double', "scalar").build()

contract = ModelContract(predict=signature)
```

At this point we can combine all our efforts into the LocalModel object. LocalModels are models before they get
 uploaded to the cluster. LocalModels are containers for all the information required to instantiate a ModelVersion
  in a Hydrosphere cluster. We'll call this model `"sqrt_model"`. 
  

Moreover, we need to specify environment in which our model will run.
Such environments are called Runtimes. You can learn more about them [here](https://hydrosphere.io/serving-docs/latest/overview/concepts.html#runtimes).
In this tutorial we will use default Python 3.7 runtime.
This runtime uses `src/func_main.py` script as an entry point, that's why we organised our files as we did.

```python
from hydrosdk.modelversion import LocalModel
from hydrosdk.image import DockerImage

sqrt_local_model = LocalModel(name="sqrt_model",
                              contract=contract,
                              runtime=DockerImage("hydrosphere/serving-runtime-python-3.7", "2.3.2", None),
                              payload=payload,
                              path=path)
```

After packing all necessary information into a LocalModel, we finally can upload it.
```python
upload_response = sqrt_local_model.upload(cluster, wait=True)
```

Let's check whether our model was successfully uploaded to the platform by looking for it.
```python
from hydrosdk.modelversion import ModelVersion
sqrt_model = ModelVersion.find(cluster, name="sqrt_model", version=1)
```

We are finished with uploading our model. Now we can get to the part where we develop a client code for our model. 

## Connect to your deployed model
 
We have uploaded our model - it's stored and versioned, but it's not running yet - we need to deploy it. To deploy a model
you should create an Application - linear pipeline of ModelVersions with monitoring and other benefits. 
You can learn more about Applications [here](https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications).

For the sake of simplicity, we'll create just a Servable - a bare deployed instance of our model version without any benefits. 
```python
from hydrosdk.servable import Servable
# There are no way right now to wait Synchronously for Servable creation, so you may need to wait a few seconds before continuing further
sqrt_servable = Servable.create(cluster, model_name="sqrt_model", version=1)
```

Servables provide [Predictor](hydrosdk/hydrosdk.predictor) objects, which should be used for data inference.
```python
predictor = sqrt_servable.predictor()
```

Predictors provide `predict` method which we can use to send our data to the model.
```python
import numpy as np

for x in range(10):
    result = predictor.predict({"x": np.random.rand()})
    print(result)
```


Now we have finished with testing our model and can safely delete the servable:
```python
Servable.delete(cluster, sqrt_servable.name)
```

In the next section we'll attach a monitoring model to this model to monitor
quality of our incoming data to prevent "thrash-in thrash out" situation.
 
 ## Attach monitoring model to your inference model

We'll create a dummy monitoring model to check inputs from our previous model. 
Similarly, we define another `func_main.py` and put it in "monitoring_model/src/monitoring_main.py".

``` code-block:: python
     :linenos:
     :caption: monitoring_model/src/func_main.py
     
     def predict(x, y):
         return {"value": float(x >= 0.5)} 
```

So our file structure will look like 

```
.
└── model
    └── src
        └── func_main.py
└── monitoring_model
    └── src
        └── func_main.py
```

And our payload is

```python
payload = ['src/func_main.py']
path = "monitoring_model/"
```

To attach one model as a metric to another model it's signature should combine both input and output of monitored model
with a single float scalar value in the output.
 ```python
signature = SignatureBuilder('predict') \
                .with_input('x', 'double', "scalar") \
                .with_intput('y', 'double', "scalar") \
                .with_output('value', 'float', "scalar").build()

contract = ModelContract(predict=signature)
```

Similarly we create a LocalModel and upload it to the cluster
```python
monitoring_model = LocalModel(name="sqrt_monitoring_model",
                         contract=contract,
                         runtime=DockerImage("hydrosphere/serving-runtime-python-3.7", "2.3.2", None),
                         payload=payload,
                         path=path)

monitoring_upload_response = monitoring_model.upload(cluster, wait=True)
```

Check that model is uploaded successfully:
```python
monitoring_model = ModelVersion.find(cluster, name="sqrt_monitoring_model", version=1)
``` 

Finally we attach this freshly uploaded model to our first one. To attach model as a metric to another model we need to:
1. Create metric configuration as a MetricSpecConfig object. In a configuration we specify id of monitoring model,
 threshold and comparison operator to compare output of monitoring model with threshold. Model considered healthy if comparison
 is `True`
2. Create new metric by calling `MetricSpec.create` and providing id of monitored model with metric config.
    
```python
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, TresholdCmpOp
metric_config = MetricSpecConfig(monitoring_model.id, 1, TresholdCmpOp.NOT_EQ)
metric_spec = MetricSpec.create(cluster, "is_greater_than_05", sqrt_model.id, metric_config)
```
We have attached monitoring model to our previously uploaded model to check input data.
All the future data we send through `sqrt_model`, together with results of inference is shadowed through all monitoring metrics.
You can explore how metrics behave in the web interface.

To simulate data we can again deploy a servable and send some data:
```python
import time
sqrt_servable = Servable.create(cluster, model_name="sqrt_model", version=1
time.sleep(20)  # Wait for servable creation

predictor = sqrt_servable.predictor()

for x in range(100):
    result = predictor.predict({"x": np.random.rand()})
    print(result)

Servable.delete(cluster, sqrt_servable.name)
```

You can explore changes on the UI.
