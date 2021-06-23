properties([
  parameters([
    choice(choices: ['patch','minor','major','tag','addon'], name: 'patchVersion', description: 'What needs to be bump?'),
    string(defaultValue:'', description: 'Force set newVersion or leave empty', name: 'newVersion', trim: false),
    string(defaultValue:'', description: 'Set grpcVersion or leave empty', name: 'grpcVersion', trim: false),
    choice(choices: ['false', 'true'], name: 'release', description: 'Release python package?'),
    choice(choices: ['local', 'global'], name: 'releaseType', description: 'It\'s local release or global?'),
   ])
])

SERVICENAME = 'hydro-serving-sdk'
SEARCHPATH = './requirements.txt'
SEARCHGRPC = 'hydro-serving-grpc'

def checkoutRepo(String repo){
  if (env.CHANGE_ID != null ){
    git changelog: false, credentialsId: 'HydroRobot_AccessToken', poll: false, url: repo, branch: env.CHANGE_BRANCH
  } else {
    git changelog: false, credentialsId: 'HydroRobot_AccessToken', poll: false, url: repo, branch: env.BRANCH_NAME
  }
}

def getVersion(){
    try{
        //remove only quotes
        version = sh(script: "export PATH=\"$HOME/.poetry/bin:$PATH\" && poetry version -s", returnStdout: true ,label: "get version").trim()
        return version
    }catch(e){
        return "file version not found" 
    }
}

def bumpVersion(String currentVersion, String newVersion, String patch, String path){
  sh("export PATH=\"$HOME/.poetry/bin:$PATH\" && poetry version prerelease")
}

def slackMessage(){
    withCredentials([string(credentialsId: 'slack_message_url', variable: 'slack_url')]) {
    //beautiful block
      def json = """
{
	"blocks": [
		{
			"type": "header",
			"text": {
				"type": "plain_text",
				"text": "$SERVICENAME: release - ${currentBuild.currentResult}!",
				"emoji": true
			}
		},
		{
			"type": "section",
			"block_id": "section567",
			"text": {
				"type": "mrkdwn",
				"text": "Build info:\n    Project: $JOB_NAME\n    Author: $AUTHOR\n    SHA: $newVersion"
			},
			"accessory": {
				"type": "image",
				"image_url": "https://res-5.cloudinary.com/crunchbase-production/image/upload/c_lpad,h_170,w_170,f_auto,b_white,q_auto:eco/oxpejnx8k2ixo0bhfsbo",
				"alt_text": "Hydrospere loves you!"
			}
		},
		{
			"type": "section",
			"text": {
				"type": "mrkdwn",
				"text": "You can see the assembly details by clicking on the button"
			},
			"accessory": {
				"type": "button",
				"text": {
					"type": "plain_text",
					"text": "Details",
					"emoji": true
				},
				"value": "Details",
				"url": "${env.BUILD_URL}",
				"action_id": "button-action"
			}
		}
	]
}
"""
    //Send message
        sh label:"send slack message",script:"curl -X POST \"$slack_url\" -H \"Content-type: application/json\" --data '${json}'"
    }
}

def bumpGrpc(String newVersion, String search, String patch, String path){
    sh script: "cat $path | grep '$search' > tmp", label: "Store search value in tmp file"
    currentVersion = sh(script: "cat tmp | cut -d'%' -f4 | sed 's/\"//g' | sed 's/,//g' | sed 's/^.*=//g'", returnStdout: true, label: "Get current version").trim()
    sh script: "sed -i -E \"s/$currentVersion/$newVersion/\" tmp", label: "Bump temp version"
    sh script: "sed -i 's/\\\"/\\\\\"/g' tmp", label: "remove quote and space from version"
    sh script: "sed -i \"s/.*$search.*/\$(cat tmp)/g\" $path", label: "Change version"
    sh script: "rm -rf tmp", label: "Remove temp file"
}

def buildPython(String command, String version){
    if(command == "build"){
      sh script:"""#!/bin/bash
          export PATH="$HOME/.poetry/bin:$PATH" &&
          poetry install &&
          poetry build
      """, label: "Build python package"
    }else if(command == "release"){
      try{
        withCredentials([usernamePassword(credentialsId: 'Hydrosphere_pypi', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
          sh script: """#!/bin/bash
              export PATH="$HOME/.poetry/bin:$PATH" &&
              poetry install &&
              poetry build &&
              poetry publish -u ${USERNAME} -p ${PASSWORD}
          """,label: "Release python package"
        }
      }catch(err){
        echo "$err"
      }
    }else{
      echo "command $command not found! Use build or release"
    }
}

//Create github release
def releaseService(String xVersion, String yVersion){
  withCredentials([usernamePassword(credentialsId: 'HydroRobot_AccessToken', passwordVariable: 'password', usernameVariable: 'username')]) {
      //Set global git
      sh script: "git diff", label: "show diff"
      sh script: "git commit --allow-empty -a -m 'Bump to $yVersion'", label: "commit to git"
      sh script: "git push https://$username:$password@github.com/Hydrospheredata/${SERVICENAME}.git --set-upstream master", label: "push all file to git"
      sh script: "git tag -a $yVersion -m 'Bump $xVersion to $yVersion version'",label: "set git tag"
      sh script: "git push https://$username:$password@github.com/Hydrospheredata/${SERVICENAME}.git --set-upstream master --tags",label: "push tag and create release"
      //Create release from tag
      sh script: "curl -X POST -H \"Accept: application/vnd.github.v3+json\" -H \"Authorization: token ${password}\" https://api.github.com/repos/Hydrospheredata/${SERVICENAME}/releases -d '{\"tag_name\":\"${yVersion}\",\"name\": \"${yVersion}\",\"body\": \"Bump to ${yVersion}\",\"draft\": false,\"prerelease\": false}'"
  }
}

node('hydrocentral') {
  try{
    stage('SCM'){
      //Set commit author
      sh script: "git config --global user.name \"HydroRobot\"", label: "Set username"
      sh script: "git config --global user.email \"robot@hydrosphere.io\"", label: "Set user email"
      checkoutRepo("https://github.com/Hydrospheredata/$SERVICENAME" + '.git')
      sh script: "curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3"
      AUTHOR = sh(script:"git log -1 --pretty=format:'%an'", returnStdout: true, label: "get last commit author").trim()
      if (params.grpcVersion == ''){
          //Set grpcVersion
          grpcVersion = sh(script: "curl -Ls https://pypi.org/pypi/hydro-serving-grpc/json | jq -r .info.version", returnStdout: true, label: "get grpc version").trim()
        }
    }

    stage('Test'){
      echo "Change id: ${env.CHANGE_ID}"
      if (env.CHANGE_ID != null){ 
        currentVersion = getVersion()
        buildPython("build", currentVersion)
      }
    }

    stage('Release'){
      echo "Change target: ${env.CHANGE_TARGET}"
      if (BRANCH_NAME == 'master' && params.release == 'true' || BRANCH_NAME == 'main' && params.release == 'true'  ){ //Not run if PR, only manual from master
          if (params.releaseType == 'global'){
              oldVersion = sh(script: "curl -Ls https://pypi.org/pypi/hydrosdk/json | jq -r .info.version", returnStdout: true, label: "get grpc version").trim()
              sh script: "echo oldVersion > version", label: "change version"
              bumpGrpc(grpcVersion,SEARCHGRPC, params.patchVersion,SEARCHPATH)
          } else {
              oldVersion = getVersion()
          }
              bumpVersion(getVersion(),params.newVersion,params.patchVersion,'version')
              newVersion = getVersion()
              buildPython("build", newVersion)
              releaseService(oldVersion, newVersion)
              buildPython("release", newVersion)
        }
      }
    //post if success
    if (params.releaseType == 'local' && params.release == 'true'){
        slackMessage()
    }
  } catch (e) {
  //post if failure
    currentBuild.result = 'FAILURE'
    if (params.releaseType == 'local' && params.release == 'true'){
        slackMessage()
    }
      throw e
  }
}
