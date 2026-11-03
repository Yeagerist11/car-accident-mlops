
pipeline {
    agent any

    environment {
        DATASET_DIR = 'D:\\car-accident-mlops\\data\\processed'
        MLFLOW_TRACKING_URI = 'sqlite:///D:/car-accident-mlops/mlflow.db'
    }

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
                    if not exist .venv\\Scripts\\python.exe (
                        py -3.12 -m venv .venv
                    )
                    .venv\\Scripts\\python.exe -m pip install --upgrade pip
                    .venv\\Scripts\\python.exe -m pip install -r requirements.txt
                '''
            }
        }

        stage('Check Dataset') {
            steps {
                bat '''
                    echo Checking training dataset...
                    if not exist "%DATASET_DIR%\\training" exit /b 1

                    echo Checking validation dataset...
                    if not exist "%DATASET_DIR%\\validation" exit /b 1

                    dir "%DATASET_DIR%\\training"
                    dir "%DATASET_DIR%\\validation"
                '''
            }
        }

        stage('Check MLflow Database') {
            steps {
                bat '''
                    if not exist "D:\\car-accident-mlops\\mlflow.db" (
                        echo MLflow database not found!
                        exit /b 1
                    )
                    echo MLflow database found.
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

        stage('Train ML Model with MLflow') {
            steps {
                echo 'Training MobileNetV2 and tracking experiment with MLflow...'

                bat '''
                    .venv\\Scripts\\python.exe -m src.train
                '''
            }
        }

        stage('Archive Trained Model') {
            steps {
                archiveArtifacts artifacts: 'models/*.keras',
                                 fingerprint: true
            }
        }
    }

    post {
        success {
            echo 'Vehicle Damage MLOps pipeline completed successfully!'
            echo 'Model training, MLflow tracking and archiving completed.'
        }

        failure {
            echo 'Pipeline failed. Please check the console output.'
        }
    }
}
