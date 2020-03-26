def repository = 'hydro-serving-sdk'

def buildAndPublishReleaseFunction={
//   need  python3-venv on Jenkins slave
//    sh "python3 -m venv venv"
//    sh "source venv/bin/activate"
    sh "pip3 install -r requirements.txt"
    sh "python3 setup.py bdist_wheel"
//    sh "pytest"   //NB(bulat): commented because there are no mocks for tests
    configFileProvider([configFile(fileId: 'PYPIDeployConfiguration', targetLocation: '.pypirc', variable: 'PYPI_SETTINGS')]) {
        sh "python3 -m twine upload --config-file ${env.WORKSPACE}/.pypirc -r pypi ${env.WORKSPACE}/dist/*"
    }
//    sh "deactivate"
}

def buildFunction={
    sh """#!/bin/bash
        python3 -m venv venv
        source venv/bin/activate
        pip install wheel~=0.34.2
        pip install -r requirements.txt
        python setup.py bdist_wheel
        #pytest
        deactivate
    """
}

def collectTestResults = {
    junit testResults: 'test-report.xml', allowEmptyResults: true
}

pipelineCommon(
    repository,
    false, //needSonarQualityGate,
    [],
    collectTestResults,
    buildAndPublishReleaseFunction,
    buildFunction,
    buildFunction
)
