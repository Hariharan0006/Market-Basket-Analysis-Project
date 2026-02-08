from flask import Flask, render_template, request
import pandas as pd
import joblib
from mlxtend.frequent_patterns import association_rules

app = Flask(__name__)

te = joblib.load("transaction_encoder.joblib")
frequent_itemsets = joblib.load("frequent_itemsets.joblib")
rules = joblib.load("association_rules.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    
    # Read uploaded CSV
    books = pd.read_csv(file)

    # Convert binary to transaction list
    transactions = []
    for i in range(len(books)):
        transactions.append(
            books.columns[books.iloc[i] == 1].tolist()
        )

    # Transform using trained encoder
    encoded = te.transform(transactions)
    df_encoded = pd.DataFrame(encoded, columns=te.columns_)

    # Generate new rules
    frequent = frequent_itemsets
    rules_df = association_rules(frequent, metric="lift", min_threshold=1)

    # Filter strong rules
    rules_df = rules_df[
        (rules_df['confidence'] >= 0.6) &
        (rules_df['lift'] > 1)
    ]

    # Convert frozenset to string
    rules_df['antecedents'] = rules_df['antecedents'].astype(str)
    rules_df['consequents'] = rules_df['consequents'].astype(str)

    return render_template(
        "result.html",
        tables=[rules_df.head(10).to_html(classes='data')],
        titles=rules_df.columns.values
    )

if __name__ == "__main__":
    app.run(debug=True)
