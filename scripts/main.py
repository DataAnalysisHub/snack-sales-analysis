import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
df_behavior = pd.read_csv('source-data/QVI_purchase_behaviour.csv')
df_transaction = pd.read_excel('source-data/QVI_transaction_data.xlsx')


# settings to display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''  Data Pre-processing   '''
# Get info about dataset
df_behavior.info()
print(df_behavior.describe())
print('customer-behavior-duplicates: ', df_behavior.duplicated().sum())         # check for duplicates
print(df_behavior.head(10), '\n\n')

df_transaction.info()
print(df_transaction.describe(), '\n')
print('transaction-duplicates: ', df_transaction.duplicated().sum(), '\n')      # check for duplicates
df_transaction = df_transaction.drop_duplicates().reset_index(drop=True)        # remove duplicates
print(df_transaction.head(10), '\n\n')


# Remove Outliers
#  Compute Z-scores
numeric_cols = ['PROD_QTY', 'TOT_SALES']
z_scores = np.abs((df_transaction[numeric_cols] - df_transaction[numeric_cols].mean()) / df_transaction[numeric_cols].std())

# Filter out data where Z-score > 3
df_transaction_cleaned = df_transaction[(z_scores < 3).all(axis=1)]

print(f"Removed {len(df_transaction) - len(df_transaction_cleaned)} outliers.")


# Sort transaction dataset by DATE column in ascending order
df_transaction = df_transaction.sort_values(by='DATE')

# convert DATE column to appropriate format
df_transaction['DATE'] = pd.to_datetime(df_transaction['DATE'], origin='1900-01-01', unit='D')

# extract year/month from date
df_transaction['TRSC_MTH'] = df_transaction['DATE'].dt.to_period('M')
df_transaction['TRSC_MTH'] = df_transaction['TRSC_MTH'].astype(str)

# extract day of week
df_transaction['DAY_OF_WEEK'] = df_transaction['DATE'].dt.day_name()

# get the unique products
print(df_transaction['PROD_NAME'].value_counts())

# remove non-chips products
df_transaction = df_transaction[~df_transaction['PROD_NAME'].str.contains("salsa", case=False)]

# remove pack sizes from product names
df_transaction['PROD_NAME'] = df_transaction['PROD_NAME'].str.replace(r'\d+\s?[a-zA-Z]*', '', regex=True)

# merge transaction data with customer behavior data
df = df_transaction.merge(df_behavior, on='LYLTY_CARD_NBR', how='left')

# remove irrelevant columns
df = df.drop(columns=['STORE_NBR', 'LYLTY_CARD_NBR', 'TXN_ID', 'PROD_NBR'])


'''  Analysis of Products and Customer behavior   '''

# 1. Calculate Total Sales
total_sales = df['TOT_SALES'].sum()


# 2. Contribution of Each Item to Monthly Total Revenue
product_revenue = (
    df.groupby(['TRSC_MTH', 'PROD_NAME'])['TOT_SALES']
    .sum()
    .reset_index()
    .rename(columns={'TOT_SALES': 'ProductRevenue'})
)
product_revenue['ContributionPercentage'] = (product_revenue['ProductRevenue'] / total_sales) * 100


# 3. Monthly Total Sales vs Qty of Purchase
monthly_sales = (
    df.groupby('TRSC_MTH').agg({
        'TOT_SALES': 'sum',
        'PROD_QTY': 'sum'
    })
)

monthly_sales.plot(kind='bar', title='Monthly Sales')
plt.xlabel('Month of Year')
plt.ylabel('Number of Purchases/Total Sales')
plt.xticks(rotation=45, ha='right', va='top')
plt.tight_layout()
plt.savefig('../figures/monthly sales.jpg')
plt.close()


# 4. Top Products Sold every Month
top_products = df.groupby(['TRSC_MTH', 'PROD_NAME'])['TOT_SALES'].sum().reset_index()
top_products = top_products.pivot(index='PROD_NAME', columns='TRSC_MTH', values='TOT_SALES')
top_products['Total'] = top_products.sum(axis=1)
top_products = top_products.nlargest(5, 'Total')
top_products = top_products.drop(columns='Total')

top_products.T.plot(kind='bar')
plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
plt.xlabel('Months/Products')
plt.ylabel('Total Sales')
plt.title(' Top 5 Monthly Purchased Products')
plt.xticks(rotation=45, ha='right', va='top')
plt.tight_layout()
plt.savefig('../figures/top-products.jpg')
plt.close()


# 5. Sales by Day of Week
sales_by_day = df.groupby('DAY_OF_WEEK')['TOT_SALES'].sum()

sales_by_day.plot(kind='bar', title='Sales by Day of Week') 
plt.xlabel('Day of Week')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right', va='top') 
plt.tight_layout() 
plt.savefig('../figures/sales-by-day.jpg')
plt.close()


# 6. Sales by Customer Class and Life Stage
customer_analysis = df.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE'])['TOT_SALES']\
    .sum()\
    .reset_index()\
    .rename(columns={'PREMIUM_CUSTOMER': 'CUSTOMER_CLASS'})

customer_analysis.to_csv('output-data/customer-analysis.csv')


# 7. Sales by Customer Class
customer_sales = (
    df.groupby('PREMIUM_CUSTOMER').agg({
        'PROD_QTY': 'sum',
        'TOT_SALES': 'sum'
    })
)

customer_sales.plot(kind='bar', title='Sales by Customer Class')
plt.xlabel('Customer Class')
plt.ylabel('Total Sales/Qty')
plt.xticks(rotation=45, ha='right', va='top')
plt.tight_layout()
plt.savefig('../figures/customer-sales.jpg')
plt.close()


# 8. Sales by Customer Life Stage
customer_lifestage = df.groupby('LIFESTAGE').agg({
        'PROD_QTY': 'sum',
        'TOT_SALES': 'sum'
    })

customer_lifestage.plot(kind='bar', title='Sales by Customer Life Stage')
plt.xlabel('Customer Life Stage')
plt.ylabel('Total Sales')
plt.xticks(rotation=45, ha='right', va='top')
plt.tight_layout()
plt.savefig('../figures/customer-lifestage.jpg')
plt.close()


# 9. average sales per transaction
df['AVG_SALES_PER_TRANSACTION'] = df['TOT_SALES']/df['PROD_QTY']


# 10. Correlation between Product Qty and Total Sales
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import zscore

# Calculate z-scores for PROD_QTY and TOT_SALES
df['Z_SCORE_PROD_QTY'] = zscore(df['PROD_QTY'])
df['Z_SCORE_TOT_SALES'] = zscore(df['TOT_SALES'])

# Filter out data points where the absolute z-score is greater than 3
filtered_df = df[
    (df['Z_SCORE_PROD_QTY'].abs() <= 3) &
    (df['Z_SCORE_TOT_SALES'].abs() <= 3)
]

# Recalculate the correlation coefficient
correlation, _ = pearsonr(filtered_df['PROD_QTY'], filtered_df['TOT_SALES'])

# Plot the scatter plot without outliers
sns.scatterplot(data=filtered_df, x='PROD_QTY', y='TOT_SALES', alpha=0.7)

# Add correlation coefficient as text to the plot
plt.text(0.05, 0.95, f"Pearson Correlation: {correlation:.2f}",
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
)

plt.title('Correlation between Product Quantity and Total Sales', fontsize=14)
plt.xlabel('Product Quantity', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../figures/correlation-plot.jpg')