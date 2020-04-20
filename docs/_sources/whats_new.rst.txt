What's new in |release|
##########################

* Predictors

.. code-block::
   :linenos:

    import pandas as pd​
    from hydrosdk.cluster import Cluster
    from hydrosdk.servable import Servable
    ​
    cluster = Cluster(http_address="https://hydro-serving.dev.hydrosphere.io",
                      grpc_address="hydro-serving.dev.hydrosphere.io")
    ​
    servable = Servable.get(cluster, "adult-classification-1-wild-flower")
    predictor = servable.predictor(ssl=True)
    ​
    data = pd.read_csv("private_hydrosphere_examples/examples/adult/data/validation.csv")
    data.columns = [s.lower().replace(" ", "_") for s in data.columns]
    ​
    for row in data.itertuples(index=False):
        prediction = predictor.predict(row._asdict())

* Bug fixes
* Other things