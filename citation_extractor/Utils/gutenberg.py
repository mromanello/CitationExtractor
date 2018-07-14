"""
A collection of print functions for debugging purpose.
Johannes Gutenberg would be proud.

"""

from __future__ import print_function

AAUTHOR = 'AAUTHOR'
AWORK = 'AWORK'
REFAUWORK = 'REFAUWORK'
NIL = 'urn:cts:GreekLatinLit:NIL'


def print_frame_with_title(title):
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


@print_frame_with_title(title='This is the title')
def print_this(this, a=0):
    print(this, a)


@print_frame_with_title(title='Dataframe distribution')
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


if __name__ == '__main__':
    print_this('ddfdf', a=3)
