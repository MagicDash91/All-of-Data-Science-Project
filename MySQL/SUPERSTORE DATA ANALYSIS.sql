SELECT * FROM new_schema.`sample - superstore - wanda.xlsx - orders`;

/* SELECT AMOUNT OF CUSTOMER EACH REGION */
SELECT Region, COUNT(Region) AS TOTAL_CUSTOMER FROM new_schema.`sample - superstore - wanda.xlsx - orders`
GROUP BY Region
ORDER BY TOTAL_CUSTOMER DESC;

/* COUNT THE QUANTITY EACH REGION */
SELECT Region, SUM(Quantity) AS TOTAL_QUANTITY FROM new_schema.`sample - superstore - wanda.xlsx - orders`
GROUP BY Region
ORDER BY TOTAL_QUANTITY DESC;

/* COUNT SALES EACH REGION */
SELECT Region, ROUND(SUM(Sales),2) AS TOTAL_SALES FROM new_schema.`sample - superstore - wanda.xlsx - orders`
GROUP BY Region
ORDER BY TOTAL_SALES DESC;

/* FIRST BUY EACH REGION */
SELECT Region, MIN(Order_Date) AS FIRST_BUYER_DATE FROM new_schema.`sample - superstore - wanda.xlsx - orders`
GROUP BY Region
ORDER BY FIRST_BUYER_DATE;