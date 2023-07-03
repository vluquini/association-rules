import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Criar a tabela de transações
dados = [
    ['não', 'sim', 'não', 'sim', 'sim', 'não', 'não'],
    ['sim', 'não', 'sim', 'sim', 'sim', 'não', 'não'],
    ['não', 'sim', 'não', 'sim', 'sim', 'não', 'não'],
    ['sim', 'sim', 'não', 'sim', 'sim', 'não', 'não'],
    ['não', 'não', 'sim', 'não', 'não', 'não', 'não'],
    ['não', 'não', 'não', 'não', 'sim', 'não', 'não'],
    ['não', 'não', 'não', 'sim', 'não', 'não', 'não'],
    ['não', 'não', 'não', 'não', 'não', 'não', 'sim'],
    ['não', 'não', 'não', 'não', 'não', 'sim', 'sim'],
    ['não', 'não', 'não', 'não', 'não', 'sim', 'não']
]

# Converter a tabela em um DataFrame
df = pd.DataFrame(dados, columns=['leite', 'café', 'cerveja', 'pão', 'manteiga', 'arroz', 'feijão'])

# Transformar os dados em transações binárias
df_encoded = pd.get_dummies(df)

# Gerar os itens frequentes usando o algoritmo Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

# Gerar as regras de associação a partir dos itens frequentes
association_rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.8)

# Exibir as regras de associação encontradas
for _, row in association_rules.iterrows():
    antecedent = set(row['antecedents'])
    consequent = set(row['consequents'])
    support = row['support']
    confidence = row['confidence']
    
    print('Antecedente:', antecedent)
    print('Consequente:', consequent)
    print('Suporte:', support)
    print('Confiança:', confidence)
    print()
