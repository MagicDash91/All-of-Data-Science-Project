SELECT C.city_id, C.city_name, S.country_name
FROM new_schema.`city` C
JOIN new_schema.`country` S
ON C.city_id = S.country_id;

SELECT C.city_id, C.city_name, S.country_name
FROM new_schema.`city` C
LEFT JOIN new_schema.`country` S
ON C.city_id = S.country_id;

SELECT C.city_id, C.city_name, S.country_name
FROM new_schema.`city` C
RIGHT JOIN new_schema.`country` S
ON C.city_id = S.country_id;