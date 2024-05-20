from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
from bs4 import BeautifulSoup
import requests

# Define default arguments
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),  # Start yesterday
    'schedule_interval': '@daily',  # Run daily
}


def scrape_wiki_content(ti):
  """
  Scrapes content from Albert Einstein's Wikipedia page and stores it in XCom.
  """
  url = 'https://en.wikipedia.org/wiki/Albert_Einstein'
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')

  # Target all paragraphs within the main content section (can be adjusted)
  content_elements = soup.find_all('p', class_=None)  # Find all paragraphs without a class

  # Combine the text content of all paragraphs
  content_text = '\n'.join([p.get_text(strip=True) for p in content_elements])

  # Store the content in XCom for retrieval by downstream tasks
  ti.xcom_push(
      key='einstein_wiki_content',
      value=content_text
  )


# Define the DAG
with DAG(
    dag_id='wiki_einstein_scraper',
    default_args=default_args,
) as dag:

    # Scrape data task
    scrape_task = PythonOperator(
        task_id='scrape_wiki_content',
        python_callable=scrape_wiki_content,  # Pass the function with TaskInstance injection
    )
