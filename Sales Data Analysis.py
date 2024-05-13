#!/usr/bin/env python
# coding: utf-8

# # Sales Data Analysis Project 

# ### Importing the dataset

# In[109]:


import pandas as pd
data = pd.read_excel('Sales_data.xlsx')


# In[110]:


data.info()


# In[111]:


data.describe()


# ### Cleaning the Dataset

# In[112]:


data['Postal Code'] = data['Postal Code'].astype('object')
data['Order ID'] = data['Order ID'].astype('object')
data['Customer ID'] = data['Customer ID'].astype('object')


# In[113]:


data.info()


# In[114]:


data.drop_duplicates(inplace=True)


# In[116]:


data.info()


# In[117]:


data.head()


# ### Data Preparation

# In[118]:


total_sales = data['Sales'].sum()
average_sales = data['Sales'].mean()

data['month_year'] = data['Order Date'].dt.to_period('M')
monthly_sales = data.groupby('month_year')['Sales'].sum()


# In[119]:


data.head()


# In[120]:


monthly_sales


# ## Data Analysis and visualization

# In[121]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index.to_timestamp(), monthly_sales.values, marker='*',color='teal',markerfacecolor='r')
plt.xlabel('\n Month')
plt.xticks(rotation=45)
plt.gca().xaxis.set_tick_params(rotation=45)

plt.ylabel(' Total Sales')
plt.title('Sales Trends Over Time')
plt.show()


# In[122]:


plt.hist(monthly_sales.values, bins=10)


# In[123]:


best_selling_products = data.groupby('Product Name')['Quantity'].sum().nlargest(10)


# In[124]:


best_selling_products


# In[125]:


plt.bar(best_selling_products.index,best_selling_products.values)
plt.xticks(rotation=90)
plt.xlabel('\n Product Name')
plt.ylabel('Total Quantity Sold')
plt.title('Top 5 Best-Selling Products')
plt.show()


# ### Calculating Average Sales per Sale

# In[126]:


total_revenue = data['Sales'].sum()
average_revenue_per_sale = data['Sales'].mean()


# In[127]:


average_revenue_per_sale


# In[128]:


revenue_growth_rate = ((monthly_sales[-1] - monthly_sales[-2]) / monthly_sales[-2]) * 100
revenue_growth_rate


# *The revenue for the current period has decreased by approximately 9.39% compared to the previous period. This decrease may be attributed to various factors such as seasonality, economic conditions, or changes in consumer behavior.*

# In[144]:


sales_by_market = data.groupby('Market')['Sales'].sum().sort_values(ascending=False)
sales_by_region = data.groupby('Region')['Sales'].sum().sort_values(ascending=False)
sales_by_segment = data.groupby('Segment')['Sales'].sum().sort_values(ascending=False)


# In[146]:


pd.options.display.float_format = '{:.2f}'.format
print(f'{sales_by_market}{sales_by_region} {sales_by_segment}')


# In[131]:


data['Profit Margin'] = (data['Profit'] / data['Sales']) * 100
data.head()


# In[132]:


data.to_excel('sales_data_updated.xlsx', index=False)


# In[133]:


product_profit_sales = data.groupby('Product Name')[['Profit', 'Sales']].sum().reset_index()

product_profit_sales['Profit Margin'] = (product_profit_sales['Profit'] / product_profit_sales['Sales']) * 100

print(product_profit_sales[['Product Name', 'Profit Margin']])


# In[134]:


sorted_product_profit_margin = product_profit_sales.sort_values(by='Profit Margin', ascending=True)

print(sorted_product_profit_margin[['Product Name', 'Profit Margin']].head(20))


# In[135]:


sorted_product_profit_margin = product_profit_sales.sort_values(by='Profit Margin', ascending=False)

print(sorted_product_profit_margin[['Product Name', 'Profit Margin']].head(20))


# In[136]:


average_revenue_customer = data.groupby('Customer Name')['Sales'].mean()


# In[137]:


sorted_product_profit_margin = average_revenue_customer.sort_values(ascending=False)

print(sorted_product_profit_margin)


# ## Findings

# * From the Line plot "Sales Trend Over Time", it is evident that the Sales increases over time and also we can see that in the beginning of each year there is a exponential decrease in the sales after which a gradual increase takes place. 

# * The best selling product is "Staples", over 800 units of Staples were sold

# * The average revenue per sale is 246.49058120257362
# 

# * The revenue for the current period has decreased by approximately 9.39% compared to the previous period. This decrease may be attributed to various factors such as seasonality, economic conditions, or changes in consumer behavior.

# * In APAC Market, Centeral regions and in Consumer segments highest revenue was analyzed

# * Products named
# ~Eureka Disposable Bags for Sanitaire Vibra Gro...
# ~Chromcraft Training Table, Adjustable Height 
# ~Bush Westfield Collection Bookcases, Dark Cher...
# ~Euro Pro Shark Stick Mini Vacuum 
# ~Chromcraft Coffee Table, Fully Assembled
# were found to have the lowest profit margin

# * Products named 
# ~Southworth Structures Collection 
# ~Xerox 1890 
# ~Tops Green Bar Computer Printout Paper
# ~Avery 475
# ~Canon imageCLASS MF7460 Monochrome Digital Las...
# were found to have the highest profit margin
# 

# * Customers named
# ~Sean Miller           
# ~Hunter Lopez        
# ~Tom Ashbrook        
# ~Christopher Conant  
# ~Mike Gockenbach 
# had the highest average revenue so they are considered as the most valuable customers.

# In[ ]:




