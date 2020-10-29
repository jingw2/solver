# Hierarchical Forecast Reconcilation Method 

## Usage
* Step 1: Input DataFrame and construct hierarchy tree 
```python
# 数据格式参考reconcilation_test
data = pd.read_csv("reconcilation_test.csv")

# get all series，获取所有品牌系列信息。all代表所有的品牌之和
series = data.loc[~data["series"].isna() & data["sku"].isna(), 
    ["series"]].drop_duplicates()
series = series["series"].tolist()
series = [s for s in series if s != "all"]
skus = data.loc[~data["sku"].isna(), ["series", "sku"]].drop_duplicates()
skus = (skus["series"] + "_" + skus["sku"]).tolist()

# top level is root, series, skus, 因为stores就1个，就作为root
total = {"root": series} # root对应层，是第一层
skus_h = {k: [v for v in skus if v.startswith(k)] for k in series}
hierarchy = {**total, **skus_h}

tree = HierarchyTree.from_nodes(hierarchy)
```
* Step 2: Split train and validation data
```python
def clear_ids(ids):
    cols = []
    for c in ids:
        if isinstance(c, tuple) or isinstance(c, list):
            cols.append(c[1])
        else:
            cols.append(c)
    new_cols = []
    for c in cols:
        if c.endswith("_"):
            if c == "all_":
                new_cols.append("root")
            else:
                new_cols.append(c[:-1])
            continue
        new_cols.append(c)
    return new_cols
    
def mape(y, ypred):
    y = np.array(y).ravel()
    ypred = np.array(ypred).ravel()
    return np.abs(y-ypred) / y

def preprocess(df):
    df.fillna("", inplace=True)
    df.loc[:, "id"] = df.loc[:, "series"] + "_" + df.loc[:, "sku"]
    df["residual"] = mape(df["y"], df["ypred"])
    return df 

train_data = data[data["flag"] == "val"] # to be changed 
val_data = data[data["flag"] == "val"]
val_data = preprocess(val_data)
train_data = preprocess(train_data)

# 预测集合, forecast data
forecasts = pd.pivot_table(val_data, values=["ypred"], index=["date"], columns=["id"])
# mape结果, MAPE result
residuals = pd.pivot_table(val_data, values=["residual"], index=["date"], columns=["id"])
# historical data to calculate ratio if using top down method
history = pd.pivot_table(train_data, values=["y"], index=["date"], columns=["id"])
forecasts.columns = clear_ids(forecasts.columns)
residuals.columns = clear_ids(residuals.columns)
history.columns = clear_ids(history.columns)
val_data["id"] = clear_ids(val_data["id"])
```
* Step 3: run recilation method
```python
res = optimal_reconcilation(forecasts, tree, method="mint", residuals=residuals)
# postprocess
res = pd.merge(res, val_data[["id", "y", "ypred", "date"]], how="left", on=["id", "date"])
res.loc[res["id"] == "root", "id"] = "all"
res["mape"] = mape(res["y"], res["ypred"])
res["mape_new"] = mape(res["y"], res["ypred_new"])
res[["series", "sku"]] = res["id"].str.split("_", expand=True)
res.drop(columns=["id"], inplace=True)
```

## Examples
To run examples, 
```shell
python reconcilation.py 
```
## Reference:
* [Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts](https://otexts.com/fpp2/).
* [Optimal Forecast Reconciliation for Hierarchical and Grouped Time
    Series Through Trace Minimization](https://robjhyndman.com/papers/MinT.pdf)
* [scikit-hts](https://github.com/jingw2/scikit-hts/blob/master/hts/functions.py)
