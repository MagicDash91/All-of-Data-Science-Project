from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from bs4 import BeautifulSoup  # For web scraping
import requests

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),  # Start yesterday
    'schedule_interval': '@daily',  # Run daily
}


def scrape_worldometer(ti):  # Inject the TaskInstance object
  """
  Scrapes Worldometer website for population data and stores in XCom.
  """
  url = 'https://www.worldometers.info/world-population/'
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  # Target elements using updated selectors
  births_today = soup.find('span', class_='rts-counter', rel='births_today').text.strip()
  deaths_today = soup.find('span', class_='rts-counter', rel='dth1s_today').text.strip()

  # Store data in XCom for retrieval by downstream tasks
  ti.xcom_push(
      key='worldometer_data',
      value={
          'births_today': births_today,
          'deaths_today': deaths_today
      }
  )

# Define the DAG
with DAG(
    dag_id='worldometer_scraper',
    default_args=default_args,
) as dag:

  # Scrape data task
  scrape_task = PythonOperator(
      task_id='scrape_worldometer',
      python_callable=scrape_worldometer,  # Pass the function with TaskInstance injection
  )
