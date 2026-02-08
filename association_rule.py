import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import warnings
warnings.filterwarnings("ignore")



books = pd.read_csv("book.csv")
books.head()

books.info()

transactions = []

for i in range(len(books)):
    transactions.append(
        books.columns[books.iloc[i] == 1].tolist()
    )

transactions[:5]


transaction_encoder = TransactionEncoder()
te_ary = transaction_encoder.fit(transactions).transform(transactions)

df_encoded = pd.DataFrame(te_ary, columns=transaction_encoder.columns_)
df_encoded.head()


item_counts = df_encoded.sum().sort_values(ascending=False)
item_counts.head(10)

item_counts.head(10).plot(kind='barh')
plt.title("Most Popular Books")
plt.show()



frequent_itemsets = apriori(
    df_encoded,
    min_support=0.05,
    use_colnames=True
)

frequent_itemsets.sort_values('support', ascending=False)

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

rules.head()

strong_rules = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

strong_rules.sort_values('lift', ascending=False)


strong_rules = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

strong_rules.sort_values('lift', ascending=False)


plt.scatter(
    strong_rules['support'],
    strong_rules['confidence'],
    c=strong_rules['lift'],
    cmap='coolwarm'
)

plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Association Rules")
plt.colorbar(label='Lift')
plt.show()

import joblib

joblib.dump(te, "transaction_encoder.joblib")
joblib.dump(frequent_itemsets, "frequent_itemsets.joblib")
joblib.dump(rules, "association_rules.joblib")


