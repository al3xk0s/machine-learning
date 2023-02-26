import pandas as pd

data = pd.read_table('pokemon.tsv')


def task_1(data):
    data = pd.read_table('pokemon.tsv')
    print(data[:10])
    print(data[-10:])
    print(data.info()) # или data.describe()


def task_2(data):
    data = data.drop(['abilities', 'japanese_name', 'type1', 'type2'], axis=1)
    print(data[:5])


def task_3(data):
    values = {
        'against_fairy': data['against_fairy'].mode()[0],
        'against_fire': data['against_fire'].max(),
        'against_grass': data['against_grass'].min(),
        'base_happiness': data['base_happiness'].mode()[0]
    }

    target_columns = list(values.keys())
    data = data.fillna(value=values)

    print(data[target_columns].isnull().sum())


def task_4(data):
    target_column = 'classfication'
    classfication_column = data[target_column]

    # Частоты уже отсортированы по убыванию
    frequence_table = classfication_column.value_counts()

    # Если в задаче подразумевается, что в случае одинаковых частот,
    # будут разные значения, то решение такое
    def get_different_columns():
        # Т.к. данные отсортированы, то в качестве нового значения
        # можно подставить index + 1
        index_gen = (i + 1 for i in range(0, len(frequence_table)))
        return frequence_table.map(lambda _: next(index_gen))

    data[target_column] = data[target_column].map(get_different_columns())

    print(data[target_column])


def task_5(data: pd.DataFrame):
    for a in [data[u].is_unique for u in data.columns]:
        print(a)

task_5(data)