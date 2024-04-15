/* CATEGORIZE FILM WITH ACTOR NAME AND FILM CATEGORY */
SELECT Z.actor_id, 
CONCAT(Z.first_name," ",Z.last_name) AS actor_name,
X.film_id, C.title AS film_title,
B.name AS category
FROM sakila.actor Z
INNER JOIN sakila.film_actor X
ON Z.actor_id = X.actor_id
INNER JOIN sakila.film_text C
ON X.film_id = C.film_id
INNER JOIN sakila.film_category V
ON C.film_id = V.film_id
INNER JOIN sakila.category B
ON V.category_id = B.category_id
WHERE B.name = 'Action';

/* CUSTOMER PAYMENT DATA WITH PRICE AND FILM TITLE */
SELECT CONCAT(Z.first_name," ",Z.last_name) AS customer_name, 
X.amount, X.payment_date,
C.inventory_id, C.rental_id,
V.film_id, B.title
FROM sakila.customer Z
INNER JOIN sakila.payment X
ON Z.customer_id = X.customer_id
INNER JOIN sakila.rental C
ON X.customer_id = C.customer_id
INNER JOIN sakila.inventory V
ON C.inventory_id = V.inventory_id
INNER JOIN sakila.film_text B
ON V.film_id = B.film_id;

/* CUSTOMER ADDRESS AND IDENTITY */
SELECT CONCAT(Z.first_name, " ", Z.last_name) AS name,
Z.email, Z.address_id,
X.address
FROM sakila.customer Z
INNER JOIN sakila.address X
ON Z.address_id = X.address_id