CREATE OR REPLACE VIEW vw_eligible_orders AS
SELECT
  order_id,
  customer_id,
  order_status,
  order_purchase_ts,
  order_approved_ts,
  order_delivered_carrier_ts,
  order_delivered_customer_ts,
  order_estimated_delivery_ts
FROM orders
WHERE lower(order_status) IN ('delivered', 'shipped', 'approved', 'invoiced');

CREATE OR REPLACE VIEW vw_fact_items AS
SELECT
  eo.order_id,
  eo.customer_id,
  c.customer_unique_id,
  c.customer_city,
  c.customer_state,
  eo.order_purchase_ts,
  eo.order_delivered_customer_ts,
  eo.order_estimated_delivery_ts,
  oi.order_item_id,
  oi.product_id,
  oi.seller_id,
  CAST(COALESCE(oi.price, 0) AS DOUBLE) AS price,
  CAST(COALESCE(oi.freight_value, 0) AS DOUBLE) AS freight_value,
  CAST(COALESCE(oi.price, 0) + COALESCE(oi.freight_value, 0) AS DOUBLE) AS line_gmv
FROM vw_eligible_orders eo
LEFT JOIN customers c ON eo.customer_id = c.customer_id
LEFT JOIN order_items oi ON eo.order_id = oi.order_id;

CREATE OR REPLACE VIEW vw_delivered_orders AS
SELECT
  order_id,
  customer_id,
  order_purchase_ts,
  order_approved_ts,
  order_delivered_carrier_ts,
  order_delivered_customer_ts,
  order_estimated_delivery_ts
FROM orders
WHERE lower(order_status) = 'delivered';
