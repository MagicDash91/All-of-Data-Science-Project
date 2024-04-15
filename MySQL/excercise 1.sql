/* no 1 */
SELECT VendorID, passenger_count, trip_distance, payment_type FROM new_schema.yellow_tlc_apr2022_1k
WHERE trip_distance < 3
AND payment_type = 3;

/* no 2 */
SELECT VendorID, passenger_count, trip_distance, payment_type FROM new_schema.yellow_tlc_apr2022_1k
WHERE trip_distance < 3;

/* no 3 */
SELECT VendorID, passenger_count, trip_distance, payment_type FROM new_schema.yellow_tlc_apr2022_1k
WHERE trip_distance < 3 
AND passenger_count = 1;

/* no 4 */
SELECT VendorID, passenger_count, trip_distance, payment_type FROM new_schema.yellow_tlc_apr2022_1k
WHERE trip_distance 
BETWEEN 1.50 AND 1.60;
