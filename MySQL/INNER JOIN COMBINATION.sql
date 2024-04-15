SELECT DISTINCT Z.customer_id, X.email, CONCAT (X.first_name,'  ', X.last_name) AS full_name , Z.inventory_id
FROM sakila.rental Z
LEFT JOIN sakila.customer X
ON Z.customer_id = X.customer_id
ORDER BY inventory_id ASC;

SELECT A.city_id, A.city, B.country 
FROM sakila.city A
INNER JOIN sakila.country B
ON A.city_id = B.country_id;

SELECT A.film_id, A.actor_id, B.category_id
FROM sakila.film_actor A
RIGHT JOIN sakila.film_category B
ON A.film_id = B.category_id;
