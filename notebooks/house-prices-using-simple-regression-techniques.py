# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import stats
from functools import partial, wraps
from itertools import combinations
from typing import Iterable, Callable, Any, TypeVar
from sklearn.preprocessing import LabelEncoder, RobustScaler
from collections import abc

# %%
pd.set_option('display.max_columns', 80)

# %%
sns.set_theme()

# %% [markdown]
# # Setup helper functions/decorators

# %%
IndexLabel = Any


# %%
def set_columns(
    df: pd.DataFrame, 
    func: Callable[[pd.DataFrame, ...], Iterable[IndexLabel]],
    *args, **kwargs) -> pd.DataFrame:
    """
    Sets column headers to result of `func`, as such result of `func` 
    may be an iterable of any types that pd.DataFrame.columns support, and must be of 
    the same length as `df`.columns.
    """
    df.columns = func(df, *args, **kwargs)
    return df


# %%
def labels(labels: Iterable[IndexLabel] | IndexLabel) -> Callable[..., Iterable[IndexLabel]]:
    """
    Silly higher-order function to play nice with functions that take a 
    Callable[[pd.DataFrame, ...], Iterable[IndexLabel]], such as set_columns.
    Use set_axis instead if you just want to set new labels in your index.
    """
    match labels:
        case str(labels):
            labels = [labels]
        case abc.Iterable() as labels:
            labels = labels
        case _:
            labels = [labels]
    def func(_) -> Iterable[IndexLabel]:
        return labels
    return func


# %%
def starassign(func: Callable[[pd.DataFrame, ...], pd.DataFrame | pd.Series]):
    """
    Decorator that **assigns result of func to `df`, which must be the first argument of the function.
    If result is a pd.Series, then assigns to column of `df` of the same name as name attribute of result. 
    """
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        res = func(df, *args, **kwargs)
        if isinstance(res, pd.Series):
            res = res.to_frame()
        return df.assign(**res)
    return wrapper


# %%
def ndarray2dataframe(func: Callable[[pd.DataFrame, ...], np.ndarray]):
    """
    Decorator that returns a dataframe of same shape as ndarray, with same columns and indices as `df`. 
    `df` must be the first argument of the function.
    """
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        res = func(df, *args, **kwargs)
        return pd.DataFrame(res, columns=df.columns, index=df.index)
    return wrapper


# %%
def operate_on_subset(
    apply_func: Callable[[pd.Series, ...], pd.Series],
    subset_func: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series | pd.core.groupby.GroupBy]
) -> Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series]:
    """
    Higher-order function that applies an operation to a subset of a DataFrame.

    Parameters:
    apply_func: Function that defines the operation to apply to the subset.
    subset_func: Function to create the subset of the DataFrame.

    Returns: A function that when called with a DataFrame and additional arguments,
    applies the operation to the subset of the DataFrame.
    """
    def func(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        subset = subset_func(df)
        return subset.apply(apply_func, *args, **kwargs)
    return func


# %%
def columns(names: Iterable[IndexLabel] | IndexLabel) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if isinstance(names, str):
        names = [names]
    def loc(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:,names]
    return loc


# %% [markdown]
# # Loading df

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Setup

# %%
dummied_cols = ['MSSubClass']
date_frag_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']

# %%
na_fill_mapping = {
    'Alley': 'NAcc',
    'BsmtQual': 'NBsmt',
    'BsmtCond': 'NBsmt',
    'BsmtExposure': 'NBsmt',
    'BsmtFinType1': 'NBsmt',
    'BsmtFinType2': 'NBsmt',
    'FireplaceQu': 'NFp',
    'GarageType': 'NGa',
    'GarageFinish': 'NGa',
    'GarageQual': 'NGa',
    'GarageCond': 'NGa',
    'PoolQC': 'NPo',
    'Fence': 'NFe',
    'MiscFeature': 'None',
    'MasVnrType': 'None',
}


# %%
def read_house_dataset(filepath, **kwargs) -> pd.DataFrame:
    """
    Reads Kaggle's house price dataset from 
    "House Prices - Advanced Regression Techniques" competittion.
    Transforms the read dataframe for my needs.
    Keyword arguments are passed to pd.read_csv.
    """
    df = (
        pd.read_csv(
            filepath, 
            index_col=0,
            **kwargs,
        )
        .fillna(na_fill_mapping)  # applying fillna so that variables where nan mean something get filled in.
        
        .astype({col:'str' for col in dummied_cols})  # converting encoded to str so that we can clearly distinguish 
                                                      # between qualitative and quantitative variables.
                                                      # decided to keep grading variables (OverallQual & OverallCond)
                                                      # as numeric so that we can draw a scatter plot.
                                                      # we will need to encode them again
    )
    return df


# %% [markdown]
# ## Loading

# %%
[f.name for f in Path('../datasets').glob('*.*')]

# %%
train_df = (
    read_house_dataset('../datasets/train.csv')
)
train_df.head()

# %%
test_df = (
    read_house_dataset('../datasets/test.csv')
)
test_df.head()

# %% [markdown]
# ## DataFrame info

# %%
display(train_df.shape, test_df.shape)

# %%
[c for c in train_df.columns if c not in test_df.columns]

# %%
display(
    train_df.dtypes.value_counts(),
    test_df.dtypes.value_counts(),
)


# %% [markdown]
# We see 9 more int64 variables and 8 less float64 in the train dataset due to nan 
# in the test dataset, see below. np.int columns are casted to np.float due to the fact
# that np.float can represent nan (IEEE 754 and all that).

# %% [markdown]
# # Descriptive stats

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Setup

# %%
def describe_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns dataframe with nan count and dtype for each column in long format in descending order.
    """
    res = (
        df.isna().sum()
        .sort_values(ascending=False)
        .to_frame(name='nancnt')
        .query('nancnt > 0')
        .assign(dtypes_=lambda df_: df.dtypes.loc[df_.index])
    )
    return res


# %% [markdown]
# ## Missing values

# %%
train_nancnt_df = describe_missing(train_df)
train_nancnt_df

# %%
test_nancnt_df = describe_missing(test_df)
test_nancnt_df

# %% [markdown]
# ## Stats

# %%
merged_df = pd.concat([train_df, test_df])

# %%
merged_df.describe(include='number')

# %%
merged_df.describe(exclude='number')


# %% [markdown]
# # Skew

# %% [markdown]
# Naive approach

# %% [markdown]
# ## Setup

# %%
@starassign
def transform(
    df: pd.DataFrame,
    func: Callable[[pd.Series, ...], pd.Series],
    subset: Callable[[pd.DataFrame], pd.DataFrame | pd.Series | pd.core.groupby.GroupBy],
    *args, **kwargs) -> pd.DataFrame:
    """
    Assigns `df` with subset of this dataframe that has had `func` applied to its columns.
    `subset` is piped into `df`.
    Positional and keyword arguments are passed to `func`.
    """
    return df.pipe(subset).apply(func, *args, **kwargs)


# %%
def high_positive_skew(df: pd.DataFrame, thresh=.75) -> pd.DataFrame:
    return df.select_dtypes(include='number').loc[:,lambda df: df.skew().gt(thresh)]


# %%
log1ptransform = partial(transform, func=np.log1p)

# %% [markdown]
# ## Visualizing skew

# %%
quantitative_df = merged_df.select_dtypes(include='number')

# %%
quantitative_df.skew().sort_values(ascending=False).plot.bar()

# %% [markdown]
# ## Transformation to normality

# %%
transformed_df = log1ptransform(quantitative_df, subset=high_positive_skew)


# %% [markdown]
# # Linear correlations

# %% [markdown]
# Naive approach

# %% [markdown]
# ## Setup

# %%
def r_pvalue(df: pd.DataFrame) -> pd.Series:
    """
    Wastefully calculates pearson's coefficient, by computing it with scipy.stats.pearsonr,
    possibly twice if you already used pd.DataFrame.corr method to get r, just to get its pvalue.
    What else can I say?
    """
    pvalue_ser = (
        df.corr(method=lambda a, b: stats.pearsonr(a, b).pvalue)
        .unstack()
    )
    return pvalue_ser


# %%
def correlate_long(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Calculates pearson's coefficient for DataFrame and returns it in long format with a multiindex. 
    Keyword args are passed to pd.DataFrame.corr method.
    """
    res = (
        df
        .corr(**kwargs)
        .unstack()
        .to_frame(name='r')
        .assign(
            r2=lambda df: df.r.pow(2),
        )
        .loc[combinations(df.columns, 2)]
    )
    return res


# %% [markdown]
# ## Correlating

# %%
corrdf = (
    transformed_df
    .pipe(correlate_long)
    .assign(pvalue=r_pvalue(transformed_df))
    .query('pvalue <= .05')
)

# %%
(
    corrdf.sort_values('r2', ascending=False)
    .head(20)
    .style.format(precision=4)
)

# %%
(
    corrdf.loc[(slice(None), 'SalePrice'),:]
    .sort_values('r2', ascending=False)
    .style.format(precision=4)
)

# %% [markdown]
# # Data distribution and relationships

# %% [markdown]
# ## Quantitative

# %% [markdown]
# ### Distribution

# %%
(
    so.Plot(transformed_df)
    .pair(x=transformed_df.columns, wrap=4)
    .add(so.Bars(alpha=1), so.Hist(stat='proportion', common_norm=False, common_bins=False))
    .share(x=False, y=False)
    .layout(size=(15, 30))
)

# %% [markdown]
# ### Direct relationships with SalePrice

# %%
(
    so.Plot(train_df.select_dtypes('number'), y='SalePrice')
    .pair(x=train_df.select_dtypes('number').columns, wrap=4)
    .add(so.Dots(), color='SalePrice')
    .layout(size=(15, 30))
)

# %% [markdown]
# ### Relations between predictors

# %%
corrdf.loc[(slice(None),'SalePrice'),:].sort_values('r2', ascending=False).head(10)

# %%
top_corr = transformed_df[['SalePrice','OverallQual','GrLivArea','GarageArea','TotRmsAbvGrd']]
sns.pairplot(top_corr, kind = 'reg', diag_kind = 'kde') 

# %% [markdown]
# ## Categorical

# %%
qualitative_df = merged_df.select_dtypes(exclude='number').astype({'MSSubClass': 'int'})

# %% [markdown]
# ### Boxplots

# %%
for col in qualitative_df.columns:
    sns.boxplot(data=qualitative_df, x=col, y=train_df.SalePrice)
    plt.show()


# %% [markdown]
# # Dealing with missing values

# %% [markdown]
# ## Setup

# %% [markdown]
# We define several functions to make it a breeze to impute data.

# %%
def filternacols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, df.isna().any()]


# %%
def fillna(ser: pd.Series, f: Callable[pd.Series, ...], *args, **kwargs) -> pd.Series:
    """
    Fills nan in pd.Series with result of `f`. Positional and keyword arguments are passed to `f`.
    """
    filling = f(ser, *args, **kwargs)
    if isinstance(filling, pd.Series) and filling.size == 1:
        filling = filling.iloc[0]
    return ser.fillna(filling)


# %%
# to be used with impute_* functions
with_mean = partial(fillna, f=pd.Series.mean)
with_mode = partial(fillna, f=pd.Series.mode)
with_median = partial(fillna, f=pd.Series.median)
# to be use inside assign. literally an alias so that code is more readable
fillna_with_mean = partial(fillna, f=pd.Series.mean)
fillna_with_mode = partial(fillna, f=pd.Series.mode)
fillna_with_median = partial(fillna, f=pd.Series.median)


# %%
@starassign
def label_encode(
    df: pd.DataFrame, 
    encoder: Callable[[pd.Series, ...], Iterable[int]] = LabelEncoder().fit_transform,
    subset: Callable[[pd.DataFrame], pd.DataFrame] = lambda df: df.select_dtypes(exclude='number'),
    *args, **kwargs) -> pd.DataFrame:
    """
    Encodes each column of the pd.DataFrame returned by `subset`. 
    `subset` is piped into `df`.
    Positional and keyword arguments are passed to `encoder`.
    Beware that default label encoder treats nan as another label. 
    """
    return df.pipe(subset).transform(encoder, *args, **kwargs)


# %%
@starassign
def impute_subset(
    df: pd.DataFrame,
    fillmethod: Callable[[pd.Series, ...], pd.Series],
    subset: Callable[[pd.DataFrame], pd.DataFrame | pd.Series | pd.core.groupby.GroupBy],
    *args, **kwargs) -> pd.DataFrame:
    """
    Imputes `df` with subset of this dataframe that has had `fillmethod` applied to its columns.
    `subset` is piped into `df`.
    If `subset` contains columns that don't have any nan, they are returned as is.
    Positional and keyword arguments are passed to `fillmethod`.
    """
    return df.pipe(subset).apply(fillmethod, *args, **kwargs)


# %%
categorical = lambda df: df.select_dtypes(exclude='number')
numerical = lambda df: df.select_dtypes(include='number')

# %%
impute_numerical = partial(
    impute_subset, 
    subset=lambda df: df.select_dtypes(include='number').pipe(filternacols),
)
impute_categorical = partial(
    impute_subset,
    subset=lambda df: df.select_dtypes(exclude='number').pipe(filternacols),
)


# %%
def groupby(
    by: Iterable[IndexLabel] | IndexLabel, 
    on: Iterable[IndexLabel] | IndexLabel,
    group_keys: bool = False, 
    *args, **kwargs) -> Callable[pd.DataFrame, pd.core.groupby.GroupBy]:
    """
    Higher-order function to create groupbys for `impute_*` functions based on `by` and `on`.
    Positional and keyword arguments are passed `df`.groupby.
    """
    def grouper(df: pd.DataFrame) -> pd.core.groupby.GroupBy:
        return df.groupby(by=by, group_keys=group_keys, *args, **kwargs)[on]
    return grouper


# %%
def impute_groupby(
    df: pd.DataFrame, 
    fillmethod: Callable[pd.Series, pd.Series], 
    by: Iterable[IndexLabel] | IndexLabel,
    on: Iterable[IndexLabel] | IndexLabel, 
    *args, **kwargs) -> pd.DataFrame:
    """
    Convenience function to impute `df` on columns `on` on groups `by`, with `fillmethod`.
    If group doesn't have any nan, they are returned as is.
    Positional and keyword arguments are passed to `fillmethod`.
    """
    grouper = groupby(by=by, on=on)
    return impute_subset(df, fillmethod, grouper, *args, **kwargs)


# %% [markdown]
# ## Taking another look at the missing values

# %%
train_nancnt_df

# %%
test_nancnt_df

# %% [markdown]
# ## MSZoning

# %% [markdown]
# Naive approach

# %%
sns.boxplot(data=merged_df, x='MSZoning', y='SalePrice')

# %%
(
    merged_df.pipe(log1ptransform, subset=high_positive_skew)
    .pipe(label_encode)
    .pipe(correlate_long)
    .loc['MSZoning',lambda df: ~df.columns.isin(['MSZoning'])]
    .sort_values('r2', ascending=False)
    .head(5)
)

# %%
pd.crosstab(*merged_df.loc[:,['Alley','MSZoning']].T.to_numpy())

# %%
sns.boxplot(data=merged_df, x='MSZoning', y='YearBuilt')

# %% [markdown]
# ## MasVnrArea

# %%
sns.boxplot(data=merged_df, x='MasVnrType', y='MasVnrArea')

# %% [markdown]
# ## BsmtBaths

# %% [markdown]
# This won't really affect the results, leaving here for reference and later feature engineering. Median of BsmtBaths is 0 anyway.

# %%
plot_data = merged_df.assign(BsmtBaths=lambda df: df.BsmtFullBath+df.BsmtHalfBath)

# %%
plot_data.BsmtBaths.median()

# %%
plot_data.groupby('BsmtQual').BsmtBaths.median()

# %%
plot_data.groupby('BsmtCond').BsmtBaths.median()

# %%
sns.boxplot(data=plot_data, x='BsmtQual', y='BsmtBaths')

# %%
sns.boxplot(data=plot_data, x='BsmtCond', y='BsmtBaths')

# %% [markdown]
# ## LotFrontage

# %% [markdown]
# Naive aproach

# %%
(
    merged_df.pipe(log1ptransform, subset=high_positive_skew)
    .pipe(label_encode)
    .pipe(correlate_long)
    .loc['LotFrontage',lambda df: ~df.columns.isin(['LotFrontage'])]
    .sort_values('r2', ascending=False)
    .head(10)
)

# %%
sns.boxplot(merged_df, x='BldgType', y='LotFrontage')

# %%
(
    so.Plot(
        merged_df, 
        x='LotArea', y='LotFrontage'
    )
    .add(so.Dots(alpha=.5), color='BldgType')
)

# %%
(
    so.Plot(
        merged_df, 
        x='LotArea', y='LotFrontage'
    )
    .facet(col='BldgType', wrap=3)
    .add(so.Dots(), color='GarageType')
    .share(x=False, y=True)
    .layout(size=(15, 10))
)

# %% [markdown]
# ### Boxplots

# %%
for col in qualitative_df.columns:
    sns.boxplot(data=qualitative_df, x=col, y=merged_df.LotFrontage)
    plt.show()

# %% [markdown]
# ## Conclusion

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Setup

# %%
# here we define a few partial functions just so it's clearer for the reader
# passing `on` as list to mute numpy's empty slice warnings
impute_lotfrontage = partial(impute_groupby, on=['LotFrontage'])
impute_masvnrarea = partial(impute_groupby, on='MasVnrArea')
impute_mszoning = partial(impute_groupby, on='MSZoning')

# %% [markdown]
# ### Imputing strategy

# %% [markdown]
# Strategy: LotFrontage by median of itself inside BldgType $\cap$ GarageType or BldgType in case there's no data for the group, MasVnrArea by median of itself inside MasVnrType, MSZoning by mode of itself inside Alley, other numerical by median, and other categorical by mode.

# %% [markdown]
# It's possible that this strategy of using the median or mode inside groups of other variables will not work as well as expected. For LotFrontage there are few observations in several groups of the intersection between the two groups.

# %%
date_frag_cols


# %%
def impute_strategy(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df
        # first we fill with median of intersection of 'BldgType' and 'GarageType'
        .pipe(impute_lotfrontage, with_median, by=['BldgType', 'GarageType'])
        # then we fill remaining with median of 'BldgType'
        .pipe(impute_lotfrontage, with_median, by='BldgType')
        .pipe(impute_masvnrarea, with_median, by='MasVnrType')
        .pipe(impute_mszoning, with_mode, by='Alley')
        .pipe(impute_categorical, with_mode)
        .pipe(impute_numerical, with_median)
        # .pipe(describe_missing)
    )
    return res


# %% [markdown]
# Since we might be introducing bias to the model by using the median of groups with few members, which would explain overfitting, we can try a second approach where we just impute LotFrontage by BldgType.

# %%
def impute_strategy2(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df
        # first we fill with median of intersection of 'BldgType' and 'GarageType'
        # .pipe(impute_lotfrontage, with_median, by=['BldgType', 'GarageType'])
        # then we fill remaining with median of 'BldgType'
        .pipe(impute_lotfrontage, with_median, by='BldgType')
        .pipe(impute_masvnrarea, with_median, by='MasVnrType')
        .pipe(impute_mszoning, with_mode, by='Alley')
        .pipe(impute_categorical, with_mode)
        .pipe(impute_numerical, with_median)
        # .pipe(describe_missing)
    )
    return res


# %% [markdown]
# # Features

# %% [markdown]
# ## Feature engineering strategy

# %% [markdown]
# I'm sure we can do better here. Maybe we could add a few categories. Not sure how to go about that.

# %%
def feature_strategy(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df.assign(
            TotalBaths=lambda df: df.loc[:,['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']].sum(axis=1),
            FullBaths=lambda df: df.loc[:,['BsmtFullBath','FullBath']].sum(axis=1),
            PorchSF=lambda df: df.loc[:, ['OpenPorchSF','EnclosedPorch',]].sum(axis=1),
            # OpenSF=lambda df: df.LotArea-(df.loc[:,['1stFlrSF','PorchSF','WoodDeckSF','PoolArea','GarageArea']].sum(axis=1)),
            TotalArea=lambda df: df.loc[:,['TotalBsmtSF','GrLivArea','GarageArea','WoodDeckSF','PorchSF']].sum(axis=1),
            LivingArea=lambda df: df.TotalBsmtSF+df.GrLivArea,
        )
    )
    return res


# %% [markdown]
# ## Scaling

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ### Setup

# %%
@starassign
def scale(
    df: pd.DataFrame, 
    scaler: Callable[[pd.DataFrame, ...], pd.DataFrame | pd.Series],
    subset: Callable[[pd.DataFrame], pd.DataFrame | pd.Series],
    *args, **kwargs) -> pd.DataFrame:
    """
    Scales each column of the pd.DataFrame returned by `subset`. 
    `subset` is piped into `df`.
    Positional and keyword arguments are passed to `scaler`.
    """
    return scaler(df.pipe(subset), *args, **kwargs)


# %% [markdown]
# ### Scaling strategy

# %%
scaling_strategy = partial(scale, scaler=ndarray2dataframe(RobustScaler().fit_transform))

# %% [markdown]
# ## Variable encoding

# %% [markdown]
# ### Setup

# %%
import json

# %%
with open('../datasets/ordinal_column_mapping.json') as f:
    ordinal_column_mapping = json.load(f)

# %%
ordinal_column_mapping


# %% [markdown]
# ### Variable encoding strategy

# %%
def label_enc_strategy(df: pd.DataFrame, mapping=ordinal_column_mapping) -> pd.DataFrame:
    res = (
        df
        .replace(mapping)  # label encode and preserve order
        .astype({k:'int64' for k in mapping.keys()})  # types in json defined as strs, so we convert them
    )
    return res


# %%
def one_hot_enc_strategy(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df.pipe(pd.get_dummies)  # one hot encode
        # one variable has spaces in the values, which doesn't play well with LightGB when we one hot encode them
        .pipe(set_columns, lambda df: df.columns.str.replace(' ', '_'))
    )
    return res


# %% [markdown]
# # Building model

# %% [markdown]
# ## Putting it all together

# %%
numerical_cols = lambda df: df.loc[:,~df.columns.isin(['OverallQual','OverallCond'])].select_dtypes(include='number')

# %%
skew_filt = lambda df: df.skew().gt(.75)

# %%
outlier_mask = lambda df: ~((df.GrLivArea > 4000) & (df.index <= 1460))

# %%
X_merged_df = (
    merged_df.drop(columns='SalePrice')
    .loc[outlier_mask]
    # here we can choose between the two impute strategies
    .pipe(impute_strategy2)
    .pipe(feature_strategy)
    .pipe(log1ptransform, subset=lambda df: numerical_cols(df).loc[:,skew_filt])
    .pipe(scaling_strategy, subset=numerical_cols)
    .pipe(label_enc_strategy)
    .pipe(one_hot_enc_strategy)
    # .select_dtypes(include='int')
    # .pipe(describe_missing)
)
X_merged_df.head()

# %% editable=true slideshow={"slide_type": ""}
X_train_df, X_test_df = (
    X_merged_df
    .pipe(lambda df: (df.loc[:1460], df.loc[1461:]))
)

# %%
# don't know if I should transform this or not
y_train = train_df.loc[outlier_mask].SalePrice.pipe(np.log1p)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## Setup

# %%
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

# %%
seed = 3995344272

# %%
linear = LinearRegression(n_jobs = -1)
lasso = Lasso(random_state = seed)
ridge = Ridge(random_state = seed)
kr = KernelRidge()
elnt = ElasticNet(random_state = seed)
dt = DecisionTreeRegressor(random_state = seed)
svm = SVR()
knn = KNeighborsRegressor(n_jobs = -1)
rf =  RandomForestRegressor(n_jobs = -1, random_state = seed)
et = ExtraTreesRegressor(n_jobs = -1, random_state = seed)
ab = AdaBoostRegressor(random_state = seed)
gb = GradientBoostingRegressor(random_state = seed)
xgb = XGBRegressor(random_state = seed, n_jobs = -1)
lgb = LGBMRegressor(random_state = seed, n_jobs = -1)

# %%
Model = BaseEstimator
FittedModel = Model


# %%
def fit(model: Model, X, y, *args, **kwargs) -> FittedModel: 
    return model.fit(X, y, *args, **kwargs)


# %%
def score(model: FittedModel, X, y, *args, **kwargs) -> float:
    return model.score(X, y, *args, **kwargs)


# %%
def predict(model: FittedModel, X, *args, **kwargs) -> np.ndarray:
    return model.predict(X)


# %%
def fit_score(model: Model, X, y, fit=fit, score=score, *args, **kwargs) -> float:
    fitted = fit(model, X, y)
    return score(fitted, X, y, *args, *kwargs)


# %% [markdown]
# ## Defining models

# %%
models = [lasso, ridge, kr, elnt, dt, svm, knn, rf, et, ab, gb, xgb, lgb]
model_labels = ['LSO', 'RIDGE', 'KR', 'ELNT', 'DT', 'SVM', 'KNN', 'RF', 'ET', 'AB', 'GB', 'XGB', 'LGB']


# %% [markdown]
# ## Train test split

# %%
def train_test_split_score(model, X, y, test_size: float):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state = seed)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    mse = mean_squared_error(prediction, Y_test)
    rmse = np.sqrt(mse)
    return rmse


# %%
tts_scored_models = [train_test_split_score(model, X_train_df, y_train, test_size=.3) for model in models]

# %%
tts_scores = pd.Series(tts_scored_models, index=model_labels)
tts_scores.plot.bar()

# %%
tts_scores.sort_values().head(5)


# %% [markdown]
# ## CV

# %%
def kfold_cv(model, X, y, k: int):
    neg_x_val_score = cross_val_score(model, X, y, cv = k, n_jobs = -1, scoring = 'neg_mean_squared_error')
    x_val_score = np.round(np.sqrt(-1*neg_x_val_score), 5)
    return x_val_score.mean()


# %%
cv10_scored_models = [kfold_cv(model, X_train_df, y_train, k=10) for model in models]

# %%
cv10_scores = pd.Series(cv10_scored_models, index=model_labels)
cv10_scores.plot.bar()

# %%
cv10_scores.sort_values().head(5)


# %% [markdown]
# ## Optimizing hyperparams

# %% [markdown]
# ### GridSearch

# %% [markdown]
# It is the simplest and I don't got time.

# %%
def grid_search_cv(model, X, y, params, k: int, verbose: int = 2):
    grid_search = GridSearchCV(
        estimator=model, 
        param_grid=params, 
        cv=k, 
        verbose=verbose,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    best_params = grid_search.best_params_ 
    best_score = np.sqrt(-1*(np.round(grid_search.best_score_, 5)))
    return best_params, best_score


# %% [markdown]
# ### Parametrization

# %% [markdown]
# #### Ridge

# %%
ridge.get_params()

# %%
ridge_params = {
    'alpha': np.arange(10, 12.01, .01),
    'random_state': [seed],
}

# %%
ridge_best_params, ridge_best_score = grid_search_cv(ridge, X_train_df, y_train, ridge_params, k=10)

# %%
display(ridge_best_params, ridge_best_score)

# %% [markdown]
# #### GB

# %%
gb.get_params()

# %%
gb_params = {
    'loss': ['squared_error', 'huber'],
    'learning_rate': [.05, .1],
    'max_features': ['sqrt'],
    'n_estimators': [1000, 2000, 3000],
    'min_samples_split': [10],
    'min_samples_leaf': [10, 15],
    'random_state': [seed],
}

# %%
gb_best_params, gb_best_score = grid_search_cv(gb, X_train_df, y_train, gb_params, k=10, verbose=3)

# %%
display(gb_best_score, gb_best_params)

# %%
# # %%timeit
# kfold_cv(gb, X_train_df, y_train, k=10)

# %% [markdown]
# #### Kernel Ridge

# %%
kr.get_params()

# %%
kr_params = {
    'alpha': np.arange(.2, .8, .05),
    'kernel': ['polynomial'], 
    'degree': [2],
    'coef0': np.arange(3, 6, .1),
}

# %%
kr_best_params, kr_best_score = grid_search_cv(kr, X_train_df, y_train, kr_params, k=10)

# %%
display(kr_best_score, kr_best_params)

# %% [markdown]
# #### SVM

# %%
svm.get_params()

# %%
svm_params = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [8, 9, 10, 11], 
    'gamma': [.001, .0008, .0009],
    'epsilon': [.01, .025, .05]
}

# %%
svm_best_params, svm_best_score = grid_search_cv(svm, X_train_df, y_train, svm_params, k=10, verbose=3)

# %%
display(svm_best_score, svm_best_params)

# %% [markdown]
# #### ElNt

# %%
elnt.get_params()

# %%
elnt_params = {
    'alpha': np.arange(.0003, .001, .00005), 
    'l1_ratio': np.arange(.80, 1, .05),
    'selection': ['cyclic', 'random'],
    'random_state': [seed],
}

# %%
elnt_best_params, elnt_best_score = grid_search_cv(elnt, X_train_df, y_train, elnt_params, k=10, verbose=3)

# %%
display(elnt_best_score, elnt_best_params)

# %% [markdown]
# ### Conclusion

# %% [markdown]
# TODO: create learning plots to visualize bias/variance

# %%
optimized_scores = cv10_scores.to_frame().T.assign(
    RIDGE=ridge_best_score,
    GB=gb_best_score,
    KR=kr_best_score,
    SVM=svm_best_score,
    ELNT=elnt_best_score,
).iloc[0]

# %%
optimized_scores.plot.bar()

# %%
optimized_scores.sort_values().head()


# %% [markdown]
# ### Submitting predicitons

# %%
def prediction_to_csv(ser: pd.Series, *args, **kwargs) -> pd.Series:
    (
        ser.rename('SalePrice')
        .to_csv(f'../submissions/{ser.name.lower()}.csv', *args, **kwargs)
    )
    return ser


# %%
def predict(model, X, y, X_test):
    model.fit(X, y)
    y_pred = np.expm1(model.predict(X_test))
    return y_pred


# %%
submission_models = {
    'RIDGE': Ridge(**ridge_best_params),
    'GB': GradientBoostingRegressor(**gb_best_params),
    'KR': KernelRidge(**kr_best_params),
    'SVM': SVR(**svm_best_params),
    'ELNT': ElasticNet(**elnt_best_params),
    'LGB': lgb,
}

# %%
predictions_df = (
    pd.DataFrame(index=X_test_df.index)
    .assign(**{model_name:predict(model, X_train_df, y_train, X_test_df) for model_name, model in submission_models.items()})
    .apply(prediction_to_csv)
)
predictions_df.head()

# %% [markdown]
# ## Ensemble

# %% [markdown]
# ### Average

# %%
(
    predictions_df.mean(axis=1)
    .rename('avg')
    .pipe(prediction_to_csv)
)

# %%
(
    predictions_df.loc[:,lambda df: ~df.columns.isin(['ELNT'])]
    .mean(axis=1)
    .rename('avg_noelnt')
    .pipe(prediction_to_csv)
)

# %% [markdown]
# ### Stacking

# %% [markdown]
# Too much work

# %% [markdown]
# # Conclusion

# %% [markdown]
# ## Impute strategy 1

# %%
pd.DataFrame([
    {'avg_noelnt': 0.12130},
    {'avg': 0.12169},
    {'elnt': 0.12787},
    {'gb': 0.12718},
    {'kr': 0.12611},
    {'lgb': 0.12833},
    {'ridge': 0.12537},
    {'svm': 0.12401},
]).stack().droplevel(0).sort_values().plot.bar()

# %% [markdown]
# ## Impute strategy 2

# %%
pd.DataFrame([
    {'avg_noelnt': 0.12081},
    {'avg': 0.12115},
    {'svm': 0.12402},
    {'ridge': 0.12535},
    {'lgb': 0.12922},
    {'kr': 0.12632},
    {'gb': 0.12365},
    {'elnt': 0.12787},
]).stack().droplevel(0).sort_values().plot.bar()

# %% [markdown]
# ## Sometimes less is more

# %% [markdown]
# Best score was 0.12081 with the simplest impute strategy.
