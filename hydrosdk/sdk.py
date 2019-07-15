import logging
import os

from hydro_serving_grpc.contract import ModelSignature, ModelContract

from hydroserving.core.monitoring.service import metric_spec_config_factory
from hydroserving.core.application.entities import ApplicationDef
from hydroserving.core.application.service import ApplicationService
from hydroserving.core.contract import field_from_dict
from hydroserving.core.image import DockerImage
from hydroserving.core.model.entities import Model as InnerModel
from hydroserving.core.model.service import ModelService
from hydroserving.core.monitoring.service import MonitoringService
from hydroserving.http.remote_connection import RemoteConnection


class Streaming:
    def __init__(self, source_topic, destination_topic, consumer_id, error_topic):
        self.error_topic = error_topic
        self.consumer_id = consumer_id
        self.source_topic = source_topic
        self.destination_topic = destination_topic
        self.compiled = {
            "sourceTopic": self.source_topic,
            "destinationTopic": self.destination_topic,
            "consumerId": self.consumer_id,
            "errorTopic": self.error_topic
        }

    def compile(self):
        return self.compiled


class Signature:
    def __init__(self, name="Predict"):
        self.name = name
        self.inputs = []
        self.outputs = []

    def with_name(self, name):
        self.name = name
        return self

    def with_input(self, name, data_type, shape, profile_type='none'):
        as_dict = {
            'name': name,
            'type': data_type,
            'shape': shape,
            'profile': profile_type
        }
        self.inputs.append(field_from_dict(name, as_dict))
        return self

    def with_output(self, name, data_type, shape, profile_type='none'):
        as_dict = {
            'name': name,
            'type': data_type,
            'shape': shape,
            'profile': profile_type
        }
        self.outputs.append(field_from_dict(name, as_dict))
        return self

    def compile(self):
        signature = ModelSignature(
            signature_name=self.name,
            inputs=self.inputs,
            outputs=self.outputs
        )
        return signature


class Monitoring:
    def __init__(self, name=None):
        self.name = name
        self.kind = None
        self.config = None
        self.health = False

    def with_health(self, health=True):
        self.health = health
        return self

    def with_spec(self, kind, **attrs):
        self.kind = kind
        self.config = metric_spec_config_factory(kind, **attrs)
        return self

    def compile(self):
        return {
            'name': self.name,
            'kind': self.kind,
            'withHealth': self.health,
            'config': self.config
        }


class Model:
    @staticmethod
    def from_existing(name, version):
        model = Model(name=name)
        model.version = version
        return model

    def __init__(self, name=None, runtime=None, host_selector=None,
                 predict=None, metadata=None, monitoring=None,
                 training_data=None, payload=None, install_command=None):
        self.name = name
        self.runtime = runtime
        self.host_selector = host_selector
        self.contract = None
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = {}
        if monitoring:
            self.monitoring = monitoring
        else:
            self.monitoring = []
        self.training_data = training_data
        self.payload = payload
        self.install_command = install_command
        self.predict = predict
        self.inner_model = None
        self.version = None

    def with_name(self, name):
        self.name = name
        return self

    def with_runtime(self, runtime):
        self.runtime = DockerImage.parse_fullname(runtime)
        return self

    def with_host_selector(self, host_selector):
        self.host_selector = host_selector
        return self

    def with_signature(self, signature):
        self.predict = signature
        return self

    def with_metadata(self, metadata):
        self.metadata = metadata
        return self

    def with_monitoring(self, monitoring):
        self.monitoring = monitoring
        return self

    def with_payload(self, payload):
        self.payload = payload
        return self

    def with_install_command(self, install_command):
        self.install_command = install_command
        return self

    def with_training_data(self, training_data_path):
        self.training_data = training_data_path
        return self

    def compile(self):
        if not self.version:
            if not self.predict:
                logging.warning("Prediction signature is not defined. Manager service will infer it.")
                self.predict = Signature()
            self.contract = ModelContract(
                model_name=self.name,
                predict=self.predict.compile()
            )
            monitors = [x.compile() for x in self.monitoring]
            self.inner_model = InnerModel(
                self.name, self.host_selector,
                self.runtime, self.contract, self.payload,
                self.training_data, self.install_command, monitors, self.metadata
            )
            self.inner_model.validate()
        return self

    def apply(self, url, enable_monitoring=True, enable_training_profile=True):
        if not self.inner_model:
            self.compile()
        if not self.version:
            conn = RemoteConnection(url)
            mons = MonitoringService(conn)
            mods = ModelService(conn, mons)
            res = mods.apply(self.inner_model, os.getcwd(), not enable_training_profile, not enable_monitoring)
            self.version = res["modelVersion"]
            return res


class PipelineServable:
    def __init__(self, model, weight):
        self.model = model
        self.weight = weight
        self.compiled = None

    def apply(self, url):
        self.model.apply(url)
        self.compiled = {
            'modelVersion': self.model.name + ":" + str(self.model.version),
            'weight': self.weight
        }
        return self.compiled


class Application:

    @staticmethod
    def singular(name, model, streaming=None):
        stage = [PipelineServable(model, 100)]
        return Application(name, [stage], streaming)

    @staticmethod
    def pipeline(name, models, streaming=None):
        return Application(name, models, streaming)

    def __init__(self, name=None, graph=None, streaming=None):
        if streaming:
            self.streaming = streaming
        else:
            self.streaming = []
        self.graph = graph
        self.name = name
        self.compiled = None

    def with_name(self, name):
        """

        Args:
            name (str): application name

        Returns:

        """
        self.name = name
        return self

    def with_model(self, model):
        """

        Args:
            model (Model): a single model

        Returns:

        """
        stage = [PipelineServable(model, 100)]
        self.graph = [stage]
        return self

    def with_pipeline(self, pipeline):
        """

        Args:
            pipeline (list of PipelineServable):

        Returns:

        """
        self.graph = pipeline
        return self

    def with_streaming(self, params):
        self.streaming = params
        return self

    def compile(self, url):
        applied_graph = {}
        stages = []
        for stage in self.graph:
            variants = []
            for service in stage:
                variants.append(service.apply(url))
            stages.append({'modelVariants': variants})
        applied_graph['stages'] = stages
        streaming = list([x.compile() for x in self.streaming])
        self.compiled = ApplicationDef(
            name=self.name,
            kafka_streaming=streaming,
            execution_graph=applied_graph
        )
        return self.compiled

    def apply(self, url):
        if not self.compiled:
            self.compile(url)
        conn = RemoteConnection(url)
        mons = MonitoringService(conn)
        mods = ModelService(conn, mons)
        apps = ApplicationService(conn, mods)
        return apps.apply(self.compiled)
