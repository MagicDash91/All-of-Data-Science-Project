SELECT gender, race_ethnicity, math_score, reading_score, writing_score, (math_score + reading_score + writing_score) AS total
FROM exam.exams
HAVING total > 200
ORDER BY total DESC;

SELECT race_ethnicity, AVG(math_score + reading_score + writing_score) AS AVERAGE
FROM exam.exams
GROUP BY race_ethnicity
ORDER BY AVERAGE DESC;

SELECT race_ethnicity, ROUND(AVG((math_score + reading_score + writing_score)/3),2) AS AVERAGE_SCORE
FROM exam.exams
GROUP BY race_ethnicity
ORDER BY AVERAGE_SCORE DESC;

SELECT race_ethnicity, ROUND(AVG(math_score),2) AS AVERAGE_MATH, ROUND(AVG(reading_score),2) AS AVERAGE_READING, ROUND(AVG(writing_score),2) AS AVERAGE_WRITING
FROM exam.exams
WHERE math_score > 70
AND reading_score > 70
AND writing_score > 70
GROUP BY race_ethnicity;

