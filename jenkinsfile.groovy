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
//    sh "python3 -m venv venv"
//    sh "source venv/bin/activate"
    sh "pip3 install -r requirements.txt"
    sh "python3 setup.py bdist_wheel"
    //    sh "pytest"   //NB(bulat): commented because there are no mocks for tests
//    sh "deactivate"
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
