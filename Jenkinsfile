pipeline {
    agent any

    environment {
        VENV        = 'venv'
        DATA_PATH   = 'data/sdss_sample.csv'
        OUTPUTS_DIR = 'outputs'
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 30, unit: 'MINUTES')
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Obteniendo el código del repositorio...'
                checkout scm
                sh 'ls -la'
            }
        }

        stage('Instalación de dependencias') {
            steps {
                echo 'Creando entorno virtual e instalando dependencias...'
                sh '''
                    python3 -m venv ${VENV}
                    . ${VENV}/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                    pip list
                '''
            }
        }

        stage('Pruebas básicas') {
            steps {
                echo 'Validando el dataset y la codificación cromosómica...'
                sh '''
                    . ${VENV}/bin/activate
                    export MPLBACKEND=Agg
                    pytest tests/ -v --junitxml=reports/junit.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                }
            }
        }

        stage('Ejecución del pipeline de AG') {
            steps {
                echo 'Ejecutando los tres módulos de Algoritmos Genéticos...'
                sh '''
                    . ${VENV}/bin/activate
                    export MPLBACKEND=Agg
                    python main.py --data ${DATA_PATH} --outputs ${OUTPUTS_DIR} 2>&1 | tee ${OUTPUTS_DIR}_run.log
                '''
            }
        }

        stage('Verificación de resultados') {
            steps {
                echo 'Comprobando que se generaron todos los artefactos esperados...'
                sh '''
                    set -e
                    for archivo in \
                        ${OUTPUTS_DIR}/metrics.json \
                        ${OUTPUTS_DIR}/fs_fitness_curve.png \
                        ${OUTPUTS_DIR}/fs_confusion_matrix.png \
                        ${OUTPUTS_DIR}/ht_fitness_curve.png \
                        ${OUTPUTS_DIR}/cl_fitness_curve.png \
                        ${OUTPUTS_DIR}/clustering_comparison.png \
                        ${OUTPUTS_DIR}/fs_fitness_history.csv \
                        ${OUTPUTS_DIR}/ht_fitness_history.csv \
                        ${OUTPUTS_DIR}/cl_fitness_history.csv
                    do
                        if [ ! -s "$archivo" ]; then
                            echo "FALTA el artefacto esperado: $archivo"
                            exit 1
                        fi
                        echo "OK: $archivo"
                    done
                '''
            }
        }
    }

    post {
        always {
            echo 'Archivando métricas, gráficas y logs de convergencia...'
            archiveArtifacts artifacts: 'outputs/**/*, *.log',
                             allowEmptyArchive: true,
                             fingerprint: true
        }
        success {
            echo 'Pipeline ejecutado correctamente.'
        }
        failure {
            echo 'El pipeline falló. Revisa el log de la etapa que se detuvo.'
        }
        cleanup {
            sh 'rm -rf ${VENV} || true'
        }
    }
}
