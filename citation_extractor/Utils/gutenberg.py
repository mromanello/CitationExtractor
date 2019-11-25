"""
A collection of print functions for debugging purpose.
Johannes Gutenberg would be proud.

"""


import json

AAUTHOR = 'AAUTHOR'
AWORK = 'AWORK'
REFAUWORK = 'REFAUWORK'
NIL = 'urn:cts:GreekLatinLit:NIL'


def with_titled_frame(title):
    def decorator(func):
        def func_wrapper(*args, **kwargs):
            padder = '='
            frame_size = 120
            header = '{left} {title} {right}'.format(left=padder * 5,
                                                     title=title,
                                                     right=padder * (frame_size - 5 - len(title) - 2))
            footer = padder * frame_size
            print(header)
            func(*args, **kwargs)
            print(footer)

        return func_wrapper

    return decorator


@with_titled_frame(title='Dataframe Distribution')
def print_df_distribution(df):
    n_aauthor = df.loc[(df['type'] == AAUTHOR) & (df['urn_clean'] != NIL)].shape[0]
    n_awork = df.loc[(df['type'] == AWORK) & (df['urn_clean'] != NIL)].shape[0]
    n_refauwork = df.loc[(df['type'] == REFAUWORK) & (df['urn_clean'] != NIL)].shape[0]
    n_nil = df.loc[df['urn_clean'] == NIL].shape[0]
    print('{}: {}'.format(AAUTHOR, n_aauthor))
    print('{}: {}'.format(AWORK, n_awork))
    print('{}: {}'.format(REFAUWORK, n_refauwork))
    print('{}: {}'.format(NIL, n_nil))
    print('Total: {}'.format(df.shape[0]))


@with_titled_frame(title='Ranking Vector')
def print_ranking_vector(vector):
    for i, (feature, weight) in enumerate(vector):
        print(i, feature, weight)


def print_candidates_comparison(list1, list2):
    pass


@with_titled_frame(title='Pretty Dict')
def print_dict(d):
    print(json.dumps(d, sort_keys=True, indent=4))
