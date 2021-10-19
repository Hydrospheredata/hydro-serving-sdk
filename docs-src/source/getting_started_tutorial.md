# Getting Started Tutorial
This tutorial will show you how to use hydrosdk. You will learn how to upload your code as a model to Hydrosphere.io,
 setup your client code to make inference, and attach a monitoring metric to your model.
 
 ``` important:: This tutorials was written for hydrosdk==$sdk_version
```
```note:: If you haven't launched Hydrosphere.io platform, please do so before proceeding with this tutorial.
 You can learn how to do by checking documentation here - https://hydrosphere.io/serving-docs/latest/install/index.html. 
```

 ## Creating a ModelVersion
 
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
from hydrosdk.signature import SignatureBuilder

signature = SignatureBuilder('infer') \
                .with_input('x', 'double', "scalar") \
                .with_output('y', 'double', "scalar") \
                .build()
```

At this point we can combine all our efforts into the [ModelVersion](hydrosdk/hydrosdk.modelversion) object using [ModelVersionBuilder](hydrosdk/hydrosdk.modelversion). We'll call this model `"sqrt_model"`.
  

Moreover, we need to specify environment in which our model will run.
Such environments are called Runtimes. You can learn more about them [here](https://docs.hydrosphere.io/about/concepts#runtimes).
In this tutorial we will use default Python 3.7 runtime.
This runtime uses `src/func_main.py` script as an entry point, that's why we organised our files as we did.

```python
from hydrosdk.modelversion import ModelVersionBuilder
from hydrosdk.image import DockerImage

sqrt_local_model_builder = ModelVersionBuilder(name="sqrt_model", path=path) \
    .with_runtime(DockerImage(name="hydrosphere/serving-runtime-python-$runtime_version", tag="$runtime_tag")) \
    .with_payload(payload) \
    .with_signature(signature)
```

After packing all necessary information into a [ModelVersionBuilder](hydrosdk/hydrosdk.modelversion), we finally can build and upload it to the cluster.
```python
sqrt_model: ModelVersion = sqrt_local_model_builder.build(cluster)
sqrt_model.lock_till_released()
```

We are finished with uploading our model. Now we can get to the part where we develop a client code for our model. 

## Connect to your deployed model
 
We have uploaded our model - it's stored and versioned, but it's not running yet - we need to deploy it. To deploy a model
you should create an Application - linear pipeline of ModelVersions with monitoring and other benefits. 
You can learn more about Applications [here](https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications).

To create a simple application with one stage we'll use [ApplicationBuilder](hydrosdk/hydrosdk.application) along with [ExecutionStageBuilder](hydrosdk/hydrosdk.application).

```python
from hydrosdk.application import ApplicationBuilder, ExecutionStageBuilder

stage = ExecutionStageBuilder().with_model_variant(model_version=sqrt_model, weight=100).build()
app_builder = ApplicationBuilder(name="sqrt_model").with_stage(stage)
sqrt_app = app_builder.build(cluster)
sqrt_app.lock_while_starting()
```

Applications provide [Predictor](hydrosdk/hydrosdk.predictor) objects, which should be used for data inference.
```python
predictor = sqrt_app.predictor()
```

Predictors provide `predict` method which we can use to send our data to the model.
```python
import numpy as np

for x in range(10):
    result = predictor.predict({"x": np.random.rand()})
    print(result)
```

Now we have finished with testing our model and can safely delete the application:
```python
from hydrosdk.application import Application

Application.delete(cluster, app_sqrt.name)
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
                .with_input('y', 'double', "scalar") \
                .with_output('value', 'float', "scalar").build()
```

Similarly we create a ModelVersion using ModelVersionBuilder and upload it to the cluster.
```python

monitoring_local_model_builder = ModelVersionBuilder(name="sqrt_monitoring_model", path=path) \
    .with_runtime(DockerImage(name="hydrosphere/serving-runtime-python-$runtime_version", tag="$runtime_tag")) \
    .with_payload(payload) \
    .with_signature(signature)

monitoring_model = monitoring_local_model_builder.build(cluster)
monitoring_model.lock_till_released()
```

Finally we attach this freshly uploaded model to our first one. To attach model as a metric to another model we need to:
1. Create metric configuration as a MetricSpecConfig object. In a configuration we specify id of monitoring model,
 threshold and comparison operator to compare output of monitoring model with threshold. Model considered healthy if comparison
 is `True`
2. Create new metric by calling `MetricSpec.create` and providing id of monitored model with metric config.
    
```python
from hydrosdk.monitoring import ThresholdCmpOp

metric = monitoring_model.as_metric(1, ThresholdCmpOp.NOT_EQ)
sqrt_model.assign_metrics([metric])
```
We have attached monitoring model to our previously uploaded model to check input data.
All the future data we send through `sqrt_model`, together with results of inference is shadowed through all monitoring metrics.
You can explore how metrics behave in the web interface.

To simulate data we can again deploy an application and send some data:
```python
sqrt_app = app_builder.build()
sqrt_app.lock_while_starting()

predictor = sqrt_app.predictor()
for x in range(100):
    result = predictor.predict({"x": np.random.rand()})
    print(result)

Application.delete(cluster, sqrt_app.name)
```

You can explore changes on the UI.

## (Internal API) Sending inference data for analysis

If your model is deployed in some other serving platform, you can utilize monitoring capabilities of Hydrosphere.

First of all, you need to register an external model:
```python
import hydrosdk
import grpc

cluster = hydrosdk.Cluster("url", "secure-grpc-url", grpc.ssl_channel_credentials())

sig_builder = hydrosdk.SignatureBuilder("predict")
sig_builder.with_input("in", "double", "scalar")
sig_builder.with_output("out", "double", "scalar")
signature = sig_builder.build()

em_builder = hydrosdk.ModelVersionBuilder("external-model-1", ".").with_signature(signature)

external_model = em_builder.build_external(cluster)
print(external_model)
```

Then, you need to prepare data and send it for analysis:
```python
from hydro_serving_grpc.serving.runtime.api_pb2 import PredictRequest, PredictResponse
from hydro_serving_grpc.serving.contract.tensor_pb2 import Tensor
from hydro_serving_grpc.serving.contract.types_pb2 import DataType

request = PredictRequest(
    inputs = {
        "in": Tensor(
            dtype = DataType.DT_DOUBLE,
            double_val = [1]
        )
    }
)
response = PredictResponse(
    outputs = {
        "out": Tensor(
            dtype = DataType.DT_DOUBLE,
            double_val = [2]

        )
    }
)
res = external_model.analyze(
    request_id = "test",
    request = request,
    response = response
)

print(res)
```