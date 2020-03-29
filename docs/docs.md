<a name=".hydrosdk"></a>
## hydrosdk

<a name=".hydrosdk.predictor"></a>
## hydrosdk.predictor

TODO: re-write predictors

<a name=".hydrosdk.monitoring"></a>
## hydrosdk.monitoring

<a name=".hydrosdk.monitoring.TresholdCmpOp"></a>
### TresholdCmpOp

```python
class TresholdCmpOp()
```

Threshold comparison operator

<a name=".hydrosdk.monitoring.MetricModel"></a>
### MetricModel

```python
class MetricModel():
 |  MetricModel(model, threshold, comparator)
```

Model having extra metric fields

<a name=".hydrosdk.monitoring.MetricSpecConfig"></a>
### MetricSpecConfig

```python
class MetricSpecConfig():
 |  MetricSpecConfig(model_version_id: int, threshold: Union[int, float], threshold_op: TresholdCmpOp, servable=None)
```

Metric specification config

<a name=".hydrosdk.monitoring.MetricSpec"></a>
### MetricSpec

```python
class MetricSpec():
 |  MetricSpec(cluster: Cluster, metric_spec_id: int, name: str, model_version_id: int, config: MetricSpecConfig)
```

Metric specification

<a name=".hydrosdk.monitoring.MetricSpec.create"></a>
#### create

```python
 | @staticmethod
 | create(cluster: Cluster, name: str, model_version_id: int, config: MetricSpecConfig)
```

Sends request to create metric spec and returns it

**Arguments**:

- `cluster`: active cluster
- `name`: 
- `model_version_id`: 
- `config`: 

**Raises**:

- `MetricSpecException`: If server returned not 200

**Returns**:

metricSpec

<a name=".hydrosdk.monitoring.MetricSpec.list_all"></a>
#### list\_all

```python
 | @staticmethod
 | list_all(cluster: Cluster)
```

Sends request and returns list with all available metric specs

**Arguments**:

- `cluster`: active cluster

**Raises**:

- `MetricSpecException`: If server returned not 200

**Returns**:

list with all available metric specs

<a name=".hydrosdk.monitoring.MetricSpec.list_for_model"></a>
#### list\_for\_model

```python
 | @staticmethod
 | list_for_model(cluster: Cluster, model_version_id: int)
```

Sends request and returns list with specs by model version

**Arguments**:

- `cluster`: active cluster
- `model_version_id`: 

**Raises**:

- `MetricSpecException`: If server returned not 200

**Returns**:

list of metric spec objs

<a name=".hydrosdk.monitoring.MetricSpec.get"></a>
#### get

```python
 | @staticmethod
 | get(cluster: Cluster, metric_spec_id: int)
```

Sends request and returns metric spec by its id

**Arguments**:

- `cluster`: active cluster
- `metric_spec_id`: 

**Raises**:

- `MetricSpecException`: If server returned not 200

**Returns**:

MetricSpec or None if nout found

<a name=".hydrosdk.monitoring.MetricSpec.delete"></a>
#### delete

```python
 | delete() -> bool
```

Deletes self (metric spec)

**Raises**:

- `MetricSpecException`: If server returned not 200

**Returns**:

result of deletion

<a name=".hydrosdk.servable"></a>
## hydrosdk.servable

<a name=".hydrosdk.servable.Servable"></a>
### Servable

```python
class Servable():
 |  Servable(cluster, model, servable_name, metadata=None)
```

Servable is an instance of a model version which could be used in application or by itself as it exposes various endpoints to your model version: HTTP, gRPC, and Kafka.
(https://hydrosphere.io/serving-docs/latest/overview/concepts.html#servable)

<a name=".hydrosdk.servable.Servable.model_version_json_to_servable"></a>
#### model\_version\_json\_to\_servable

```python
 | @staticmethod
 | model_version_json_to_servable(mv_json: dict, cluster)
```

Deserializes model version json to servable object

**Arguments**:

- `mv_json`: model version json
- `cluster`: active cluster

**Returns**:

servable object

<a name=".hydrosdk.servable.Servable.create"></a>
#### create

```python
 | @staticmethod
 | create(cluster, model_name, model_version, metadata=None)
```

Sends request to server and returns servable object

**Arguments**:

- `cluster`: 
- `model_name`: 
- `model_version`: 
- `metadata`: 

**Raises**:

- `ServableException`: If server returned not 200

**Returns**:

servable

<a name=".hydrosdk.servable.Servable.get"></a>
#### get

```python
 | @staticmethod
 | get(cluster, servable_name)
```

Sends request to server and return servable object by name

**Arguments**:

- `cluster`: active cluster
- `servable_name`: 

**Raises**:

- `ServableException`: If server returned not 200

**Returns**:

servable

<a name=".hydrosdk.servable.Servable.list"></a>
#### list

```python
 | @staticmethod
 | list(cluster)
```

Sends request to server and returns list of all servables

**Arguments**:

- `cluster`: active cluster

**Returns**:

json with request result

<a name=".hydrosdk.servable.Servable.delete"></a>
#### delete

```python
 | @staticmethod
 | delete(cluster, servable_name)
```

Sends request to delete servable by name

**Arguments**:

- `cluster`: 
- `servable_name`: 

**Raises**:

- `ServableException`: If server returned not 200

**Returns**:

json response from server

<a name=".hydrosdk.model"></a>
## hydrosdk.model

<a name=".hydrosdk.model.resolve_paths"></a>
#### resolve\_paths

```python
resolve_paths(path, payload)
```

Appends each element of payload to the path and makes {resolved_path: payload_element} dict

**Arguments**:

- `path`: absolute path
- `payload`: list of relative paths

**Returns**:

dict with {resolved_path: payload_element}

<a name=".hydrosdk.model.read_yaml"></a>
#### read\_yaml

```python
read_yaml(path)
```

Deserializes LocalModel from yaml definition

**Arguments**:

- `path`: 

**Raises**:

- `InvalidYAMLFile`: if passed yamls are invalid

**Returns**:

LocalModel obj

<a name=".hydrosdk.model.Metricable"></a>
### Metricable

```python
class Metricable():
 |  Metricable()
```

Every model can be monitored with a set of metrics (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#metrics)

<a name=".hydrosdk.model.Metricable.as_metric"></a>
#### as\_metric

```python
 | as_metric(threshold: int, comparator: MetricSpec) -> MetricModel
```

Turns model into Metric Model

**Arguments**:

- `threshold`: 
- `comparator`: 

**Returns**:

MetricModel

<a name=".hydrosdk.model.Metricable.with_metrics"></a>
#### with\_metrics

```python
 | with_metrics(metrics: list)
```

Adds metrics to the model

**Arguments**:

- `metrics`: list of metrics

**Returns**:

self Metricable

<a name=".hydrosdk.model.LocalModel"></a>
### LocalModel

```python
class LocalModel(Metricable):
 |  LocalModel(name, contract, runtime, payload, path=None, metadata=None, install_command=None)
```

Local Model (A model is a machine learning model or a processing function that consumes provided inputs and produces predictions or transformations, https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

<a name=".hydrosdk.model.LocalModel.create"></a>
#### create

```python
 | @staticmethod
 | create(path, name, runtime, contract=None, install_command=None)
```

Validates contract

**Arguments**:

- `path`: 
- `name`: 
- `runtime`: 
- `contract`: 
- `install_command`: 

**Returns**:

None

<a name=".hydrosdk.model.LocalModel.model_json_to_upload_response"></a>
#### model\_json\_to\_upload\_response

```python
 | @staticmethod
 | model_json_to_upload_response(cluster, model_json, contract, runtime)
```

Deserialize model json into UploadResponse object

**Arguments**:

- `cluster`: 
- `model_json`: 
- `contract`: 
- `runtime`: 

**Returns**:

UploadResponse obj

<a name=".hydrosdk.model.LocalModel.from_file"></a>
#### from\_file

```python
 | @staticmethod
 | from_file(path)
```

Reads model definition from .yaml file or serving.py

**Arguments**:

- `path`: 

**Raises**:

- `ValueError`: If not yaml or py

**Returns**:

LocalModel obj

<a name=".hydrosdk.model.LocalModel.upload"></a>
#### upload

```python
 | upload(cluster) -> dict
```

Uploads Local Model

**Arguments**:

- `cluster`: active cluster

**Raises**:

- `MetricSpecException`: If model not uploaded yet

**Returns**:

{model_obj: upload_resp}

<a name=".hydrosdk.model.Model"></a>
### Model

```python
class Model(Metricable):
 |  Model(id, name, version, contract, runtime, image, cluster, metadata=None, install_command=None)
```

Model (A model is a machine learning model or a processing function that consumes provided inputs and produces predictions or transformations, https://hydrosphere.io/serving-docs/latest/overview/concepts.html#models)

<a name=".hydrosdk.model.Model.find"></a>
#### find

```python
 | @staticmethod
 | find(cluster, name, version)
```

Finds a model on server by name and model version

**Arguments**:

- `cluster`: active cluster
- `name`: model name
- `version`: model version

**Raises**:

- `Exception`: if server returned not 200

**Returns**:

Model obj

<a name=".hydrosdk.model.Model.find_by_id"></a>
#### find\_by\_id

```python
 | @staticmethod
 | find_by_id(cluster, model_id)
```

Finds a model on server by id

**Arguments**:

- `cluster`: active cluster
- `model_id`: model id

**Raises**:

- `Exception`: if server returned not 200

**Returns**:

Model obj

<a name=".hydrosdk.model.Model.delete_by_id"></a>
#### delete\_by\_id

```python
 | @staticmethod
 | delete_by_id(cluster, model_id)
```

Deletes model by id

**Arguments**:

- `cluster`: active cluster
- `model_id`: model id

**Returns**:

if 200, json. Otherwise None

<a name=".hydrosdk.model.Model.list_models"></a>
#### list\_models

```python
 | @staticmethod
 | list_models(cluster)
```

List all models on server

**Arguments**:

- `cluster`: active cluster

**Returns**:

json-response from server

<a name=".hydrosdk.model.Model.to_proto"></a>
#### to\_proto

```python
 | to_proto()
```

Turns Model to Model version

**Returns**:

model version obj

<a name=".hydrosdk.model.ExternalModel"></a>
### ExternalModel

```python
class ExternalModel():
 |  ExternalModel(name, id_, contract, version, metadata)
```

External models running outside of the Hydrosphere platform (https://hydrosphere.io/serving-docs/latest/how-to/monitoring-external-models.html)

<a name=".hydrosdk.model.ExternalModel.ext_model_json_to_ext_model"></a>
#### ext\_model\_json\_to\_ext\_model

```python
 | @staticmethod
 | ext_model_json_to_ext_model(ext_model_json: dict)
```

Deserializes external model json to external model

**Arguments**:

- `ext_model_json`: external model json

**Returns**:

external model obj

<a name=".hydrosdk.model.ExternalModel.create"></a>
#### create

```python
 | @staticmethod
 | create(cluster, name: str, contract: ModelContract, metadata: Optional[dict] = None)
```

Creates external model on the server

**Arguments**:

- `cluster`: active cluster
- `name`: name of ext model
- `contract`: 
- `metadata`: 

**Raises**:

- `Exception`: If server returned not 200

**Returns**:

external model

<a name=".hydrosdk.model.ExternalModel.find_by_name"></a>
#### find\_by\_name

```python
 | @staticmethod
 | find_by_name(cluster, name, version)
```

Finds ext model on server by name and version

**Arguments**:

- `cluster`: active cluster
- `name`: 
- `version`: 

**Returns**:

Model

<a name=".hydrosdk.model.ExternalModel.delete_by_id"></a>
#### delete\_by\_id

```python
 | @staticmethod
 | delete_by_id(cluster, model_id)
```

Deletes external model by model id

**Arguments**:

- `cluster`: active cluster
- `model_id`: 

**Returns**:

None

<a name=".hydrosdk.model.BuildStatus"></a>
### BuildStatus

```python
class BuildStatus(Enum)
```

Model building statuses

<a name=".hydrosdk.model.UploadResponse"></a>
### UploadResponse

```python
class UploadResponse():
 |  UploadResponse(model, version_id)
```

Received status from server about uploading

<a name=".hydrosdk.model.UploadResponse.logs"></a>
#### logs

```python
 | logs()
```

Sends request, saves and returns logs iterator

**Returns**:

log iterator

<a name=".hydrosdk.model.UploadResponse.set_status"></a>
#### set\_status

```python
 | set_status() -> None
```

Checks last log record and sets upload status

**Raises**:

- `StopIteration`: If something went wrong with iteration over logs

**Returns**:

None

<a name=".hydrosdk.model.UploadResponse.get_status"></a>
#### get\_status

```python
 | get_status()
```

Gets current status of upload

**Returns**:

status

<a name=".hydrosdk.model.UploadResponse.not_ok"></a>
#### not\_ok

```python
 | not_ok() -> bool
```

Checks current status and returns if it is not ok

**Returns**:

if not uploaded

<a name=".hydrosdk.model.UploadResponse.ok"></a>
#### ok

```python
 | ok() -> bool
```

Checks current status and returns if it is ok

**Returns**:

if uploaded

<a name=".hydrosdk.model.UploadResponse.building"></a>
#### building

```python
 | building() -> bool
```

Checks current status and returns if it is building

**Returns**:

if building

<a name=".hydrosdk.application"></a>
## hydrosdk.application

<a name=".hydrosdk.application.streaming_params"></a>
#### streaming\_params

```python
streaming_params(in_topic, out_topic)
```

Deserializes topics into StreamingParams

**Arguments**:

- `in_topic`: input topic
- `out_topic`: output topic

**Returns**:

StreamingParams

<a name=".hydrosdk.application.Application"></a>
### Application

```python
class Application():
 |  Application(name, execution_graph, kafka_streaming, metadata)
```

An application is a publicly available endpoint to reach your models (https://hydrosphere.io/serving-docs/latest/overview/concepts.html#applications)

<a name=".hydrosdk.application.Application.app_json_to_app_obj"></a>
#### app\_json\_to\_app\_obj

```python
 | @staticmethod
 | app_json_to_app_obj(application_json)
```

Deserializes json into Application

**Arguments**:

- `application_json`: input json with application object fields
:return Application : application object

<a name=".hydrosdk.application.Application.list_all"></a>
#### list\_all

```python
 | @staticmethod
 | list_all(cluster)
```

Lists all available applications from server

**Arguments**:

- `cluster`: active cluster

**Raises**:

- `Exception`: If response from server is not 200

**Returns**:

deserialized list of application objects

<a name=".hydrosdk.application.Application.find_by_name"></a>
#### find\_by\_name

```python
 | @staticmethod
 | find_by_name(cluster, app_name)
```

By the *app_name* searches for the Application

**Arguments**:

- `cluster`: active cluster

**Raises**:

- `Exception`: If response from server is not 200

**Returns**:

deserialized Application object

<a name=".hydrosdk.application.Application.delete"></a>
#### delete

```python
 | @staticmethod
 | delete(cluster, app_name)
```

By the *app_name* deletes Application

**Arguments**:

- `cluster`: active cluster

**Raises**:

- `Exception`: If response from server is not 200

**Returns**:

response from the server

<a name=".hydrosdk.application.Application.create"></a>
#### create

```python
 | @staticmethod
 | create(cluster, application: dict)
```

By the *app_name* searches for the Application

**Arguments**:

- `cluster`: active cluster
- `application`: dict with necessary to create application fields

**Raises**:

- `Exception`: If response from server is not 200

**Returns**:

deserialized Application object

<a name=".hydrosdk.application.Application.parse_streaming_params"></a>
#### parse\_streaming\_params

```python
 | @staticmethod
 | parse_streaming_params(in_list: List[dict]) -> list
```

Deserializes from input list StreamingParams

**Arguments**:

- `in_list`: input list of dicts

**Returns**:

list ofr StreamingParams

<a name=".hydrosdk.application.Application.parse_singular_app"></a>
#### parse\_singular\_app

```python
 | @staticmethod
 | parse_singular_app(in_dict)
```

Part of parse_application method, parses singular

**Arguments**:

- `in_dict`: singular def

**Returns**:

stages with model variants

<a name=".hydrosdk.application.Application.parse_singular"></a>
#### parse\_singular

```python
 | @staticmethod
 | parse_singular(in_dict)
```

Part of parse_application method, parses singular pipeline stage

**Arguments**:

- `in_dict`: pipieline stage

**Returns**:

model version id and weight

<a name=".hydrosdk.application.Application.parse_model_variant_list"></a>
#### parse\_model\_variant\_list

```python
 | @staticmethod
 | parse_model_variant_list(in_list) -> list
```

Part of parse_application method, parses list of model variants

**Arguments**:

- `in_list`: dict with list model variants

**Returns**:

list of services

<a name=".hydrosdk.application.Application.parse_model_variant"></a>
#### parse\_model\_variant

```python
 | @staticmethod
 | parse_model_variant(in_dict)
```

Part of parse_application method, parses model variant

**Arguments**:

- `in_dict`: dict with model variant

**Returns**:

dict with model version and weight

<a name=".hydrosdk.application.Application.parse_pipeline_stage"></a>
#### parse\_pipeline\_stage

```python
 | @staticmethod
 | parse_pipeline_stage(stage_dict)
```

Part of parse_application method, parses pipeline stages

**Arguments**:

- `stage_dict`: dict with list of pipeline stages

**Returns**:

dict with list of model variants

<a name=".hydrosdk.application.Application.parse_pipeline"></a>
#### parse\_pipeline

```python
 | @staticmethod
 | parse_pipeline(in_list)
```

Part of parse_application method, parses pipeline

**Arguments**:

- `in_list`: input list with info about pipeline

**Returns**:

dict with list of pipeline stages

<a name=".hydrosdk.application.Application.parse_application"></a>
#### parse\_application

```python
 | @staticmethod
 | parse_application(in_dict)
```

Deserializes received from yaml file dict into Application Definition

**Arguments**:

- `in_dict`: received from yaml file dict

**Raises**:

- `ValueError`: If wrong definitions are provided

**Returns**:

Application Definition

<a name=".hydrosdk.host_selector"></a>
## hydrosdk.host\_selector

TODO: not used

<a name=".hydrosdk.cluster"></a>
## hydrosdk.cluster

<a name=".hydrosdk.cluster.Cluster"></a>
### Cluster

```python
class Cluster()
```

Cluster responsible for server interactions

<a name=".hydrosdk.cluster.Cluster.connect"></a>
#### connect

```python
 | @staticmethod
 | connect(address)
```

The preferable factory method for Cluster creation. Use it.

**Arguments**:

- `address`: connection address

**Raises**:

- `ConnectionError`: if no connection with cluster

**Returns**:

cluster object

Checks the address, the connectivity, and creates an instance of Cluster.

<a name=".hydrosdk.cluster.Cluster.__init__"></a>
#### \_\_init\_\_

```python
 | __init__(address)
```

Cluster ctor. Don't use it unless you understand what you are doing.

<a name=".hydrosdk.cluster.Cluster.request"></a>
#### request

```python
 | request(method, url, **kwargs)
```

Formats address and url, send request

**Arguments**:

- `method`: type of request
- `url`: url for a request to be sent to
- `kwargs`: additional args

**Returns**:

request res

<a name=".hydrosdk.cluster.Cluster.grpc_secure"></a>
#### grpc\_secure

```python
 | grpc_secure(credentials=None, options=None, compression=None)
```

Validates credentials and returns grpc secure channel

**Arguments**:

- `credentials`: 
- `options`: 
- `compression`: 

**Returns**:

grpc secure channel

<a name=".hydrosdk.cluster.Cluster.grpc_insecure"></a>
#### grpc\_insecure

```python
 | grpc_insecure(options=None, compression=None)
```

Returns grpc insecure channel

**Arguments**:

- `options`: 
- `compression`: 

**Returns**:

grpc insecure channel

<a name=".hydrosdk.cluster.Cluster.build_info"></a>
#### build\_info

```python
 | build_info()
```

Returns manager, gateway, sonar builds info

**Returns**:

manager, gateway, sonar build infos

<a name=".hydrosdk.cluster.Cluster.safe_buildinfo"></a>
#### safe\_buildinfo

```python
 | safe_buildinfo(url)
```

**Arguments**:

- `url`: 

**Raises**:

- `ConnectionError`: if no connection
- `JSONDecodeError`: if json is not properly formatted

**Returns**:

result request json

<a name=".hydrosdk.exceptions"></a>
## hydrosdk.exceptions

<a name=".hydrosdk.exceptions.ServableException"></a>
### ServableException

```python
class ServableException(BaseException)
```

Servable class base exception

<a name=".hydrosdk.exceptions.MetricSpecException"></a>
### MetricSpecException

```python
class MetricSpecException(BaseException)
```

Metric Spec base exception

<a name=".hydrosdk.errors"></a>
## hydrosdk.errors

<a name=".hydrosdk.errors.InvalidYAMLFile"></a>
### InvalidYAMLFile

```python
class InvalidYAMLFile(RuntimeError):
 |  InvalidYAMLFile(file, msg)
```

Error raised when working with invalid yaml file

<a name=".hydrosdk.contract"></a>
## hydrosdk.contract

<a name=".hydrosdk.contract.ContractViolationException"></a>
### ContractViolationException

```python
class ContractViolationException(Exception)
```

Exception raised when contract is violated

<a name=".hydrosdk.contract.field_from_dict"></a>
#### field\_from\_dict

```python
field_from_dict(name, data_dict)
```

Old version of deserialization *data_dict* into ModelField. Should not be used or tested first

<a name=".hydrosdk.contract.field_from_dict_new"></a>
#### field\_from\_dict\_new

```python
field_from_dict_new(name, data_dict)
```

Deserialization of *data_dict* into ModelField.

**Arguments**:

- `name`: name of passed data
- `data_dict`: data

**Raises**:

- `ValueError`: If data_dict is invalid

**Returns**:

ModelField

<a name=".hydrosdk.contract.signature_to_dict"></a>
#### signature\_to\_dict

```python
signature_to_dict(signature: ModelSignature)
```

Serializes signature into signature_name, inputs, outputs

**Arguments**:

- `signature`: model signature obj

**Raises**:

- `TypeError`: If signature invalid

**Returns**:

dict with signature_name, inputs, outputs

<a name=".hydrosdk.contract.contract_to_dict"></a>
#### contract\_to\_dict

```python
contract_to_dict(contract: ModelContract)
```

Serializes model contract into model_name, predict

**Arguments**:

- `contract`: model contract

**Returns**:

dict with model_name, predict

<a name=".hydrosdk.contract.field_to_dict"></a>
#### field\_to\_dict

```python
field_to_dict(field: ModelField)
```

Serializes model field into name, profile and optional shape

**Arguments**:

- `field`: model field

**Raises**:

- `TypeError`: If field is invalid

**Returns**:

dict with name and profile

<a name=".hydrosdk.contract.attach_ds"></a>
#### attach\_ds

```python
attach_ds(result_dict, field)
```

Adds dtype or subfields

**Arguments**:

- `result_dict`: 
- `field`: 

**Raises**:

- `ValueError`: If field invalid

**Returns**:

result_dict with dtype or subfields

<a name=".hydrosdk.contract.shape_to_dict"></a>
#### shape\_to\_dict

```python
shape_to_dict(shape)
```

Serializes model field's shape to dict

**Arguments**:

- `shape`: TensorShapeProto

**Returns**:

dict with dim and unknown rank

<a name=".hydrosdk.contract.contract_from_dict_yaml"></a>
#### contract\_from\_dict\_yaml

```python
contract_from_dict_yaml(data_dict)
```

Old version of deserialization of yaml dict into model contract. Should not be used or tested first

**Arguments**:

- `data_dict`: contract from yaml

**Returns**:

model contract

<a name=".hydrosdk.contract.contract_from_dict"></a>
#### contract\_from\_dict

```python
contract_from_dict(data_dict)
```

Deserialization of yaml dict into model contract

**Arguments**:

- `data_dict`: contract from yaml

**Returns**:

model contract

<a name=".hydrosdk.contract.parse_field"></a>
#### parse\_field

```python
parse_field(name, dtype, shape, profile=ProfilingType.NONE)
```

Deserializes into model field

**Arguments**:

- `name`: name of model field
- `dtype`: data type of model field
- `shape`: shape of model field
- `profile`: profile of model field

**Raises**:

- `ValueError`: If dtype is invalid

**Returns**:

model field obj

<a name=".hydrosdk.contract.SignatureBuilder"></a>
### SignatureBuilder

```python
class SignatureBuilder():
 |  SignatureBuilder(name)
```

Build Model Signature

<a name=".hydrosdk.contract.SignatureBuilder.with_input"></a>
#### with\_input

```python
 | with_input(name, dtype, shape, profile=ProfilingType.NONE)
```

Adds input to the SignatureBuilder

**Arguments**:

- `name`: 
- `dtype`: 
- `shape`: 
- `profile`: 

**Returns**:

self SignatureBuilder

<a name=".hydrosdk.contract.SignatureBuilder.with_output"></a>
#### with\_output

```python
 | with_output(name, dtype, shape, profile=ProfilingType.NONE)
```

Adds output to the SignatureBuilder

**Arguments**:

- `name`: 
- `dtype`: 
- `shape`: 
- `profile`: 

**Returns**:

self SignatureBuilder

<a name=".hydrosdk.contract.SignatureBuilder.build"></a>
#### build

```python
 | build()
```

Creates Model Signature

**Returns**:

ModelSignature obj

<a name=".hydrosdk.contract.AnyDimSize"></a>
### AnyDimSize

```python
class AnyDimSize(object)
```

Validation class for dimensions

<a name=".hydrosdk.contract.AnyDimSize.__eq__"></a>
#### \_\_eq\_\_

```python
 | __eq__(other)
```

If dimension is of Number type than equal

**Arguments**:

- `other`: dimension

**Raises**:

- `TypeError`: If other not Number

**Returns**:



<a name=".hydrosdk.contract.are_shapes_compatible"></a>
#### are\_shapes\_compatible

```python
are_shapes_compatible(a, b)
```

Compares if shapes are compatible

**Arguments**:

- `a`: 
- `b`: 

**Returns**:

result of comparision as bool

<a name=".hydrosdk.contract.are_dtypes_compatible"></a>
#### are\_dtypes\_compatible

```python
are_dtypes_compatible(a, b, strict=False)
```

Compares if data types are compatible

**Arguments**:

- `a`: 
- `b`: 
- `strict`: 

**Returns**:

result of comparision as bool

<a name=".hydrosdk.contract.validate"></a>
#### validate

```python
validate(t, strict=False)
```

Return bool whether array is valid for this field and error message, if not valid.
Error message is None if array is valid.

**Arguments**:

- `strict`: Strict comparison for dtypes.
- `t`: input Tensor

**Returns**:

is_valid, error_message

<a name=".hydrosdk.contract.mock_input_data"></a>
#### mock\_input\_data

```python
mock_input_data(signature: ModelSignature)
```

Creates dummy input data

**Arguments**:

- `signature`: 

**Returns**:

list of input tensors

<a name=".hydrosdk.data"></a>
## hydrosdk.data

<a name=".hydrosdk.data.conversions"></a>
## hydrosdk.data.conversions

TODO: whole file is not used

<a name=".hydrosdk.data.types"></a>
## hydrosdk.data.types

<a name=".hydrosdk.image"></a>
## hydrosdk.image

