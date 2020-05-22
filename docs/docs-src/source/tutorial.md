# Sample tutorial
This tutorial will show you how to use hydrosdk.You will learn how to upload your code as a model to Hydrosphere.io,
 attach a monitoring metric to it and setup your client code to make inference.
 ``` important:: This tutorials is written for hydrosdk~=2.3
```
```note:: If you haven't launched Hydrosphere.io platform, please do so before proceeding with this tutorial. You can learn how to do it here - https://hydrosphere.io/serving-docs/latest/install/index.html. 
```
 ## Upload your code as a Local Model
 
First we need to connect to the Hydrosphere.io platform by creating [Cluster](hydrosdk/hydrosdk.cluster) object.
 ```python
from hydrosdk.cluster import Cluster
cluster = Cluster("your-hydrosphere-cluster-address")
```

Next we need to write the script which will be uploaded to the Hydrosphere platform. For the simplicity of this tutorial let's imagine
that our model will be calculating square root of a double value provided.

We'll call this script 'func_main.py' and put it in "model/src/func_main.py".

``` code-block:: python
     :linenos:
     :caption: model/src/func_main.py

     from math import sqrt
        
     def infer(x):
        return sqrt(x)
```


Our file structure at this point should look like this:

```
.
└── model
    └── src
        └── func_main.py
```

To let hydrosdk know which files needs to be uploaded to the Hydrosphere.io platform we will capture all 
necessary file paths in a `payload` variable.
```python
payload = ['model/src/func_main.py']
```

Hydrosphere Serving is a strictly typed inference engine, so before uploading our model we need to specify it's signature with
`SignatureBuilder`.
```python
from hydrosdk.contract import SignatureBuilder, ModelContract

signature = SignatureBuilder('infer') \
                .with_input('input_', 'double', "scalar") \
                .with_output('output', 'double', "scalar").build()
contract = ModelContract(predict=signature)
```

At this point we can combine all our efforts into LocalModel object. We'll call it `"sqrt_model"`. Moreover, we need to
specify environment in which our model will run. Such environments are called Runtimes. You can learn more about them [here](https://hydrosphere.io/serving-docs/latest/overview/concepts.html#runtimes).
In this tutorial we will use default Python 3.6 runtime. This runtime uses `src/func_main.py` script as an entry point, that's why
we organised our files as we did.
```python
from hydrosdk.modelversion import LocalModel
from hydrosdk.image import DockerImage

sqrt_local_model = LocalModel(name="sqrt_model",
                              contract=contract,
                              runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
                              payload=payload)
```

After packing all necessary information into LocalModel, we finally can upload it.
```python
upload_response = sqrt_local_model._LocalModel__upload(cluster)
```

Let's check whether our model was successfuly uploaded to the platform by looking for it by id returned in `upload_response`.
```python
from hydrosdk.modelversion import ModelVersion
sqrt_model = ModelVersion.find_by_id(cluster, upload_response.modelversion.id)
```

We are finished with uploading our model. In the next section we will attach monitoring model to this model to monitor
 quality of our incoming data to prevent "thrash-in thrash out" situation. 

 ## Attach monitoring model to your inference model
 
Similarly, we define another `func_main.py` and put it in "monitoring_model/src/monitoring_main.py".

``` code-block:: python
     :linenos:
     :caption: monitoring_model/src/func_main.py

     # return 1 if our `input_` is non-negative and 0 otherwise.
     def predict(input_, output):
         return int(input_ >= 0) 
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
payload = ['monitoring_model/src/func_main.py']
```

To attach one model as a metric to another model it's signature should combine both input and output of monitored model
with a single scalar value in the output.
 ```python
signature = SignatureBuilder('predict') \
                .with_input('input_', 'double', "scalar") \
                .with_intput('output', 'double', "scalar") \
                .with_output('value', 'double', "scalar").build()
contract = ModelContract(predict=signature)
```

Similarly we create a LocalModel and upload it to Hydrosphere.io
```python
monitoring_model = LocalModel(name="multiplication_model",
                         contract=contract,
                         runtime=DockerImage("hydrosphere/serving-runtime-python-3.6", "2.1.0", None),
                         payload=payload)

monitoring_upload_response = monitoring_model._LocalModel__upload(cluster)
```

Check that model is uploaded successfully:
```python
monitoring_model = ModelVersion.find_by_id(cluster, monitoring_upload_response.modelversion.id)
``` 

Finally we attach this freshly uploaded model to our first one. To attach model as a metric to another model we need to
1. Create metric configuration as a MetricSpecConfig object. In configuration we specify id of monitoring model,
 threshold and comparison operator to compare output of monitoring model with threshold. Model considered healthy if comparison
 is `True`
2. Create new metric by calling `MetricSpec.create` and providing id of monitored model with metric config.
    
```python
from hydrosdk.monitoring import MetricSpec, MetricSpecConfig, TresholdCmpOp
metric_config = MetricSpecConfig(monitoring_model.id, 1, TresholdCmpOp.NOT_EQ)
metric_spec = MetricSpec.create(cluster, "test", sqrt_model.id, metric_config)
```

We have attached monitoring model to our previously uploaded model to check input data. Now we can get to the part 
where we develop a client code for our model. 

 ## Connect to your deployed model 
We have uploaded our model - it's stored and versioned, but it's not running yet - we need to deploy it.
To send data through our model we need create an instance of our model version. Such instances are called Servables.

```python
from hydrosdk.servable import Servable
sqrt_servable = Servable.create(cluster, model_name=sqrt_model.name, version=sqrt_model.version)
```

Servables provide [Predictor](hydrosdk/hydrosdk.predictor) objects, which should be used for data inference.
```python
predictor = sqrt_servable.predictor()
```

Predictors provide `predict` method which we can use to send data through.
```python
import random
data = random.sample(range(-10,1000), 100)

for x in data:
    result = predictor.predict({"in": x})
    print(result)
```

All the data we sent through `sqrt_model`, together with results of inference is shadowed through all monitoring metrics.
You can explore how metrics behave in the web interface. 