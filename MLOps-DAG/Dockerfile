# Use official Airflow image
FROM apache/airflow:2.6.3-python3.9

# Set environment variables for Airflow
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False

# Copy your DAGs and requirements
COPY dags/ $AIRFLOW_HOME/dags/
COPY requirements.txt $AIRFLOW_HOME/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r $AIRFLOW_HOME/requirements.txt

# Initialize DB and trigger the DAG automatically
# Run scheduler and webserver in background
CMD airflow db init && \
    airflow scheduler & \
    airflow webserver --port 8080 &
    airflow dags trigger Boston