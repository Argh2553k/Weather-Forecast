import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

local_weather = pd.read_csv('local_weather.csv')
local_weather = local_weather.set_index("DATE")

core_weather = local_weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()

del core_weather["SNOW"]
del core_weather["SNWD"]

core_weather["PRCP"] = core_weather["PRCP"].fillna(0)
core_weather = core_weather.fillna(method = "ffill")

core_weather.index = pd.to_datetime(core_weather.index)

core_weather["target"] = core_weather.shift(-1)["TMAX"]
core_weather = core_weather.iloc[:-1,:].copy()

model = Ridge(alpha = .1)
predictors = ["PRCP", "TMAX", "TMIN"]

train_set = core_weather.loc[:"2020-12-31"]
test_set = core_weather.loc["2021-01-01":]

model.fit(train_set[predictors], train_set["target"])

predictions = model.predict(test_set[predictors])

print(mean_absolute_error(test_set["target"], predictions))