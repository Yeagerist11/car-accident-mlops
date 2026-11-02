
pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                echo 'Getting project from GitHub...'
                checkout scm
            }
        }

        stage('Check Python') {
            steps {
                bat 'py -3.12 --version'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat '''
                    py -3.12 -m venv .venv
                    .venv\\Scripts\\python.exe -m pip install --upgrade pip
                    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
                '''
            }
        }

        stage('Check Python Syntax') {
            steps {
                bat '.venv\\Scripts\\python.exe -m compileall -q src api tests'
            }
        }

        stage('Run Tests') {
            steps {
                bat '.venv\\Scripts\\python.exe -m pytest tests/ -v'
            }
        }
    }

    post {
        success {
            echo 'Vehicle Damage MLOps pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Please check the console output.'
        }
    }
}
