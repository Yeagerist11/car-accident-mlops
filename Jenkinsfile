
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

        stage('Check Python Syntax') {
            steps {
                bat 'py -3.12 -m compileall -q src api'
                echo 'Python syntax check completed.'
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
