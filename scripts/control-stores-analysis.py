import pandas as pd

# load data
df = pd.read_csv('../source-data/QVI_data.csv')

# Get details about data
df.info()
print('df.describe()')
print('data-duplicates: ', df.duplicated().sum(), '\n')      # check for duplicates


'''  Data Pre-processing   '''

df = df.drop_duplicates().reset_index(drop=True)        # remove duplicates

# convert DATE column to appropriate format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d-%m-%y')

# Extract month and year from DATE
df['YearMonth'] = df['DATE'].dt.to_period('M')

# Compute monthly sales metrics
  # First calculate the unique customer count for each store and time period
num_customers = df.groupby(['STORE_NBR', 'YearMonth'])['LYLTY_CARD_NBR'].nunique()

  # Then use the agg function for the rest of your aggregations
sales_stores = df.groupby(['STORE_NBR', 'YearMonth']).agg(
    total_sales=('TOT_SALES', 'sum'),
    transactions=('TXN_ID', 'count')
).reset_index()

  # Merge the unique customer count back into the result
sales_stores = sales_stores.merge(num_customers.rename('num_customers'), on=['STORE_NBR', 'YearMonth'])

# Compute transactions per customer
sales_stores['txns_per_customer'] = sales_stores['transactions']/sales_stores['num_customers']

# Filter dataset to obtain ONLY pre-trial period (before Feb 2019)
pre_trial_sales = sales_stores[sales_stores['YearMonth'] < '2019-02']

# Compute average values of stores sales
stores_avg = pre_trial_sales.groupby('STORE_NBR').agg(
    avg_total_sales=('total_sales', 'mean'),
    avg_num_customers=('num_customers', 'mean'),
    avg_txns_per_customer=('txns_per_customer', 'mean')
)

# Finding the best control stores
import numpy as np
from scipy.stats import pearsonr

def find_best_correlation(trial_store, metrics=['total_sales', 'num_customers', 'transactions']):
    correlations = {}

    for store in pre_trial_sales['STORE_NBR'].unique():
        if store == trial_store:
            continue            # skip if it's any of the trial stores

        store_corrs = []        # store correlation values for the metrics

        for metric in metrics:
            # Get year/month for the trial and the control store
            trial_series = pre_trial_sales[pre_trial_sales['STORE_NBR'] == trial_store][['YearMonth', metric]].set_index('YearMonth')
            control_series = pre_trial_sales[pre_trial_sales['STORE_NBR'] == store][['YearMonth', metric]].set_index('YearMonth')

            # Find common dates
            common_dates = trial_series.index.intersection(control_series.index)

            if len(common_dates) > 1:
                # Convert DataFrame to 1D array
                trial_values = trial_series.loc[common_dates].values.flatten()
                control_values = control_series.loc[common_dates].values.flatten()

                # Skip if all values are the same (constant array)
                if np.all(trial_values == trial_values[0]) or np.all(control_values == control_values[0]):
                    continue       # Skip this store, as Pearson correlation is undefined (requires two entries that are NOT the same)

                corr, _ = pearsonr(trial_values, control_values)
                store_corrs.append(corr)

        # Compute average correlation for the metrics
        if store_corrs:
            correlations[store] = np.mean(store_corrs)

    # Find the store with the highest average correlation (if any exist)
    if correlations:
        best_store = max(correlations, key=correlations.get)
        return best_store, correlations[best_store]
    else:
        return None

# Function call
trial_stores = [77, 86, 88]
control_stores = {store: find_best_correlation(store)[0] for store in trial_stores}

print('control-stores: ', control_stores)


'''  Comparing Control stores vs Trial Stores during Trial period   '''

# Define the trial period (Feb 2019 - Apr 2019)
trial_period = (sales_stores['YearMonth'] >= '2019-02') & (sales_stores['YearMonth'] <= '2019-04')

# Get trial and control store data
trial_data = sales_stores[sales_stores['STORE_NBR'].isin(trial_stores) & trial_period]
control_data = sales_stores[sales_stores['STORE_NBR'].isin(control_stores.values()) & trial_period]

# Create columns to label respective stores
trial_data['Store_Type'] = 'Trial'
control_data['Store_Type'] = 'Control'

# Merge trial and control data into a single DataFrame
comparison_data = pd.concat([trial_data, control_data])

# Aggregate data by store type (Trial vs Control)
summary_stats = comparison_data.groupby(['Store_Type', 'YearMonth']).agg(
    total_sales=('total_sales', 'sum'),
    num_customers=('num_customers', 'sum'),
    avg_txns_per_customer=('txns_per_customer', 'mean')
).reset_index()

print(summary_stats)


'''  Visualise performance of Control & Trial Stores   '''

import matplotlib.pyplot as plt
import seaborn as sns

# Convert YearMonth to String before plotting
summary_stats['YearMonth'] = summary_stats['YearMonth'].astype(str)

# Sales
plt.figure(figsize=(12, 6))
sns.lineplot(data=summary_stats, x='YearMonth', y='total_sales', hue='Store_Type', marker='o')

plt.title("Total Sales: Trial vs Control Stores (Feb-Apr 2019)")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45, ha='right', va='top')
plt.legend(title="Store Type")
plt.tight_layout()
plt.savefig('../figures/control-vs-trial-sales.jpg')
plt.close()

# Number of Customers
plt.figure(figsize=(12, 6))
sns.lineplot(data=summary_stats, x='YearMonth', y='num_customers', hue='Store_Type', marker='o')

plt.title("Number of Customers: Trial vs Control Stores (Feb-Apr 2019)")
plt.xlabel("Month")
plt.ylabel("Total Customers")
plt.xticks(rotation=45, ha='right', va='top')
plt.legend(title="Store Type")
plt.tight_layout()
plt.savefig('../figures/control-vs-trial-customers.jpg')
plt.close()

# Transactions per Customer
plt.figure(figsize=(12, 6))
sns.lineplot(data=summary_stats, x='YearMonth', y='avg_txns_per_customer', hue='Store_Type', marker='o')

plt.title("Avg Transactions per Customer: Trial vs Control Stores (Feb-Apr 2019)")
plt.xlabel("Month")
plt.ylabel("Avg Transactions per Customer")
plt.xticks(rotation=45, ha='right', va='top')
plt.legend(title="Store Type")
plt.tight_layout()
plt.savefig('../figures/control-vs-trial-transactions.jpg')
plt.close()


'''  Perform statistical testing (T-Test) to gauge sales between Control & Trial Stores   '''

from scipy.stats import ttest_ind

# Extract sales for trial vs control Stores
trial_sales = trial_data.groupby('YearMonth')['total_sales'].sum()
control_sales = control_data.groupby('YearMonth')['total_sales'].sum()

# Perform T-test
t_stat, p_value = ttest_ind(trial_sales, control_sales, equal_var=False)

print(f'T-test results: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}')

# Check if the difference is significant
if p_value < 0.05:
    print('✅ The difference in sales between trial and control stores is statistically significant')
else:
    print('❌ No significant difference in sales between trial and control stores')

