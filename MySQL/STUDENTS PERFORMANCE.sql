SELECT * FROM new_schema.students_performance_mv;

/* COUNT RACE ETHNICITY WHERE test preparation course is completed and ORDER BY ASCENDING */
SELECT race_ethnicity, COUNT(race_ethnicity) AS TOTAL FROM new_schema.students_performance_mv
WHERE test_preparation_course = 'completed'
GROUP BY race_ethnicity
ORDER BY TOTAL;

/* COUNT THE TOTAL SCORE EACH STUDENT AND THEN RANK THEM FROM HIGHEST*/
SELECT gender, race_ethnicity, test_preparation_course, math_score + reading_score + writing_score AS TOTAL_SCORE
FROM new_schema.students_performance_mv
ORDER BY TOTAL_SCORE DESC;

/* COUNT THE AVERAGE SCORE OF 3 TEST THEN COUNT THE AVERAGE AGAIN GROUP BY RACE ETHNICITY THEN ELIMINATE NULL VALUE AND test preparation course is completed */
SELECT race_ethnicity, (AVG(math_score + reading_score + writing_score)/3) AS NILAI_3_PELAJARAN_RATA_RATA
FROM new_schema.students_performance_mv
WHERE test_preparation_course = 'completed'
AND NOT race_ethnicity =''
GROUP BY race_ethnicity
ORDER BY NILAI_3_PELAJARAN_RATA_RATA DESC
