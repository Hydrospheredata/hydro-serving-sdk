def repository = 'hydro-serving-sdk'

def buildAndPushDocs = {
     sh """#!/bin/bash
        python3 -m venv docs_venv
        source docs_venv/bin/activate
        python setup.py install
        pip install -r docs/docs-src/requirements.txt
        make -C docs/docs-src github
        git add docs/*
        git commit -m "Documentation Rebuilt"
        git push
        deactivate
    """
    }


def buildAndPublishReleaseFunction = {

    configFileProvider([configFile(fileId: 'PYPIDeployConfiguration', targetLocation: '.pypirc', variable: 'PYPI_SETTINGS')]) {
        sh """#!/bin/bash
        python3 -m venv venv
        source venv/bin/activate
        pip install wheel~=0.34.2
        pip install -r requirements.txt
        python setup.py bdist_wheel
        #pytest
        python -m twine upload --config-file ${env.WORKSPACE}/.pypirc -r pypi ${env.WORKSPACE}/dist/*
    """
    }

    buildAndPushDocs()
}

def buildFunction = {
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
