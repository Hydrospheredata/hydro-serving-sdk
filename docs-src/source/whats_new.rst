What's new in |release|
##########################

Features:
* ModelVersion: revoked LocalModel, useing ModelVersionBuild instead
* Signature: using Signature instead of Contract
* Cluster: added flag to skip connection check
* Deployment Configuration: switched from custom dataclass to pydantic dataclass
* Monitoring: added ability to configure monitoring configuration

Fixes:
* Cluster: simplified constructor
* Cluster: simplified urljoin function
* Servable: stabilized Servable's lock_while_starting method
* Model: fixed model's path resolover
* ModelVersion: fix training data path resolver