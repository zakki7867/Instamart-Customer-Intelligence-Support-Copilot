-- queries.sql: Analytical SQL Queries for Part A

-- Q1: WHERE then HAVING
-- Keep only completed orders with order_amount > 500 (WHERE), group by customer,
-- and keep customers whose total of those orders exceeds 8000 (HAVING).
-- Reference: 30 customers.
-- @query: Q1
SELECT 
    customer_id, 
    ROUND(SUM(order_amount), 2) AS total_amount
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_amount DESC;


-- Q2: Three-table join
-- Revenue by city and product category (join orders, customers, products).
-- Reference: 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45.
-- @query: Q2
SELECT 
    c.city, 
    p.category, 
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Scalar subquery
-- Products priced above the average catalogue price.
-- Reference: 11 products.
-- @query: Q3
SELECT 
    product_id, 
    product_name, 
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- Q4: LEFT JOIN
-- Customers who have no completed order.
-- Reference: 25 customers.
-- @query: Q4
SELECT 
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5: Common Table Expression (CTE)
-- One CTE computes each customer's total completed spend and a second computes the average;
-- returns customers whose total is above the average.
-- Reference: 550 customers; average INR 2,552.06.
-- @query: Q5
WITH cust_spend AS (
    SELECT 
        customer_id, 
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT 
        AVG(total_spend) AS avg_total_spend
    FROM cust_spend
)
SELECT 
    cs.customer_id, 
    cs.total_spend
FROM cust_spend cs, avg_spend
WHERE cs.total_spend > avg_spend.avg_total_spend
ORDER BY cs.total_spend DESC;


-- Q6: Ranking window functions
-- For each customer with completed spend, compute total spend and then
-- RANK(), DENSE_RANK() and ROW_NUMBER() using OVER (PARTITION BY city ORDER BY total_spend DESC).
-- Reference: Top customer per city: Bengaluru C0082, Chennai C0930, Delhi C0291,
-- Hyderabad C0743, Mumbai C0693, Pune C0426; exactly 8 rows have RANK() != ROW_NUMBER().
-- @query: Q6
WITH cust_spend AS (
    SELECT 
        o.customer_id, 
        c.city, 
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY o.customer_id, c.city
)
SELECT 
    customer_id, 
    city, 
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_val,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_val,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM cust_spend
ORDER BY city, rank_val;


-- Q7: Running total
-- Daily completed revenue (first 10 characters of order_ts) with cumulative revenue
-- computed as SUM(revenue) OVER (ORDER BY day).
-- Reference: 183 daily rows; last cumulative value INR 2,998,668.80.
-- @query: Q7
WITH daily_rev AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS day, 
        ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day, 
    revenue,
    ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_rev
ORDER BY day;
