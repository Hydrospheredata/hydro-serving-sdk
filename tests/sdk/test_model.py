import json
import unittest

import requests
import requests_mock

from hydrosdk.sdk import Model, Signature, Application, Monitoring

DEV_ENDPOINT = "http://localhost"

class ModelSpec(unittest.TestCase):
    def test_model_creation(self):
        monitoring = [
            Monitoring('test').with_health(True).with_spec('LatencyMetricSpec', interval=15),
            Monitoring('acc').with_health(True).with_spec("Accuracy")
        ]
        res = [x.compile() for x in monitoring]
        print(res)

        signature = Signature('predict')\
            .with_input('in1', 'float32', [-1], 'text')\
            .with_output('out1', 'int32', 'scalar', 'numerical')

        model = Model()\
            .with_name("sdk-model")\
            .with_runtime("hydrosphere/serving-runtime-python-3.6:dev")\
            .with_payload(['/Users/bulat/Documents/dev/hydrosphere/serving/example/models/claims_model/src'])\
            .with_monitoring(monitoring)\
            .with_signature(signature)
        print(model)
        model.compile()
        print(model.inner_model.__dict__)

    def test_model_apply(self):
        monitoring = [
            Monitoring('test').with_health(True).with_spec('LatencyMetricSpec', interval=15),
            Monitoring('acc').with_health(True).with_spec("Accuracy")
        ]
        signature = Signature().with_name("claim")\
            .with_input("client_profile", "float64", [112], "numerical")\
            .with_output("amount", "int64", "scalar", "real")

        model = Model() \
            .with_name("sdk-model") \
            .with_runtime("hydrosphere/serving-runtime-python-3.6:dev") \
            .with_payload(['/Users/bulat/Documents/dev/hydrosphere/serving/example/models/claims_model/src'])\
            .with_signature(signature)\
            .with_monitoring(monitoring)
        result = model.apply(DEV_ENDPOINT)
        print(model.inner_model.__dict__)
        print(result)

    def test_singular_application_apply(self):
        # def matcher(req):
        #     print(req.text)
        #     resp = requests.Response()
        #     resp.status_code = 200
        #     resp._content = json.dumps({'id': 1}).encode("utf-8")
        #     return resp
        # with requests_mock.Mocker() as mock:
        #     mock.add_matcher(matcher)
        monitoring = [
            Monitoring('sdk-ae').with_health(True).with_spec('AEMetricSpec', threshold=10, application="autoencoder", input="client_profile")
        ]
        signature = Signature().with_name("claim") \
            .with_input("client_profile", "float64", [112], "numerical") \
            .with_output("amount", "int64", "scalar", "real") \

        model = Model() \
            .with_name("sdk-model") \
            .with_runtime("hydrosphere/serving-runtime-python-3.6:dev") \
            .with_payload(['/Users/bulat/Documents/dev/hydrosphere/serving/example/models/claims_model/src']) \
            .with_signature(signature)\
            .with_monitoring(monitoring)

        app = Application.singular("sdk-app", model)
        result = app.apply(DEV_ENDPOINT)
        print(app.compiled)
        print(result)
