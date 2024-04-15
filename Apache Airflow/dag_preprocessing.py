from airflow import DAG
from airflow.providers.mysql.hooks.mysql import MySqlHook
from datetime import datetime
from airflow.operators.python import PythonOperator

# Replace with your actual connection ID
connection_id = 'mysql'

def test_mysql_connection():
    try:
        # Get the connection from Airflow
        mysql_hook = MySqlHook(mysql_conn_id=connection_id)

        # Attempt a simple connection test (e.g., ping the server)
        with mysql_hook.get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM marketing.customer;")
            result = cursor.fetchone()

        if result:
            print("Connection to MySQL successful!")
        else:
            print("Connection test failed!")

    except Exception as e:
        print(f"Error connecting to MySQL: {e}")

with DAG(dag_id='test_mysql_connection',
          start_date=datetime(2024, 4, 15),
          schedule_interval=None) as dag:

    test_connection_task = PythonOperator(
        task_id='test_connection',
        python_callable=test_mysql_connection
    )
