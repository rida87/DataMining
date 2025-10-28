
import pandas as pd  # For data manipulation
from mlxtend.preprocessing import TransactionEncoder  # For encoding transaction data
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth  # For association rule mining

transactions = pd.read_csv('store_data_association_rules.csv')  # Load dataset

#transactions.to_csv('new_transacrions.csv', sep=';', index=False)  

transactions.columns = [f'item{i+1}' for i in range(transactions.shape[1])]  # Rename columns 

print('first row before encoding')
print(transactions.iloc[0])  # Display the first transaction before encoding

transactions_list = transactions.apply(lambda row: row.dropna().tolist(), axis=1).tolist()  # Convert rows to lists, drop NaN

print('first item of the list')
print(transactions_list[0])  # Show first transaction as list

te = TransactionEncoder()  # Initialize encoder
te_ary = te.fit(transactions_list).transform(transactions_list)  # Fit and transform data to one-hot format
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)  # Create DataFrame with encoded items

df_encoded.to_csv('df_encoded.csv', sep=';', index=False)  # Optional: save encoded data

frequent_itemsets_apriori = apriori(df_encoded, min_support=0.05, use_colnames=True)  # Find frequent itemsets using Apriori
frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.05, use_colnames=True)  # Find frequent itemsets using FP-Growth

rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1)  # Generate rules from Apriori
rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1)  # Generate rules from FP-Growth

rules_apriori = rules_apriori.sort_values(by="lift", ascending=False)  # Sort Apriori rules by lift
rules_fpgrowth = rules_fpgrowth.sort_values(by="lift", ascending=False)  # Sort FP-Growth rules by lift

print('apriori')
print(rules_apriori.head())  # Display top Apriori rules

print('fpgrowth')
print(rules_fpgrowth.head())  # Display top FP-Growth rules


