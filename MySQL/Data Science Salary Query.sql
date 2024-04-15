/* SELECT DATA WHERE SALARY > 100000 */
SELECT * FROM new_schema.ds_salaries
WHERE salary > 100000;

/* SELECT DATA WHERE SALARY > 100000, Company location in US, Order the salary from the largest */
SELECT MyUnknownColumn, job_title, salary_in_usd, company_location FROM new_schema.ds_salaries
WHERE salary_in_usd > 100000
AND company_location = 'US'
ORDER BY salary_in_usd DESC;

/* Count the average Average Salary in USD group by job title and sort from the largest */
SELECT AVG(salary_in_usd) AS AVERAGE_SALARY_IN_USD, job_title FROM new_schema.ds_salaries
GROUP BY job_title
ORDER BY AVERAGE_SALARY_IN_USD DESC;