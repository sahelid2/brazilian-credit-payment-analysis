#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


cust_file = 'https://bitbucket.org/vishal_derive/vcu-data-mining/src/master/data/olist_customers_dataset.csv'
orders_file = 'https://bitbucket.org/vishal_derive/vcu-data-mining/src/master/data/olist_orders_dataset.csv'
orderpay_file ='https://bitbucket.org/vishal_derive/vcu-data-mining/src/master/data/olist_order_payments_dataset.csv'


# In[7]:


def read_olist_data(file1, file2, verbose):
    
    # read the orders data
    orders = pd.read_csv(file1)

    if verbose:
        print (f'{len(orders):,} read from the orders file.')

    # drop unnecessary columns
    drop_vars = ['order_approved_at', 'order_delivered_carrier_date', 
                 'order_delivered_customer_date', 'order_estimated_delivery_date']

    orders = orders.drop(drop_vars, axis=1)

    # date-time conversion
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    # let's convert the order purchase timestamps into dates
    orders['order_purchase_date'] = orders['order_purchase_timestamp'].dt.date

    # extract month from the order date
    orders['order_dow'] = orders['order_purchase_timestamp'].dt.weekday_name

    # read the file that contains the unique customer identifier
    # also, let's keep only the following two columns: customer_id, customer_unique_id
    cust = pd.read_csv(file2, usecols=['customer_id', 'customer_unique_id'])
    
    if verbose:
        print (f'{len(cust):,} read from the customer file.')

    # merge orders and cust dataframes
    orders_out = pd.merge(orders, cust, on='customer_id', how='inner')
    
    # apply filters to (a) discard (incomplete) data after 2018-8-22; see 06_pandas_wrangle.ipynb for the rationale
    #  and (b) keep 'delivered' orders only
    #  we do this here by using a boolean (True/False) mask
    mask = (orders_out['order_purchase_date'] <= date(2018, 8, 22)) & (orders_out['order_status'] == 'delivered')

    orders_out = orders_out[mask]
    
    # discard 'order_status' as we don't need it any more
    orders_out = orders_out.drop('order_status', axis=1)
    
    # let's keep only those columns that we need (for this exercise)
    keep_cols = ['customer_unique_id', 'order_id', 'order_purchase_timestamp', 'order_dow']

    orders_out = orders_out[keep_cols].sort_values(['customer_unique_id', 'order_purchase_timestamp'])

    if verbose:
        print (f'{len(orders_out):,} records in the output  file.')
    
    return orders_out


# In[ ]:




