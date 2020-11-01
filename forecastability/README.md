# Forecastability Analysis 

This is a tool to implement forecastability analysis, including calculating:
* Frequency 
* Stability 
* Periodicity
* Percent of Products that single customer occupies over 50% demands

## Input Data
| columname  | type  | note  | 
|---|---|---|
| date  | string  | yyyy-mm-dd or yyyy/mm/dd  |
| sku_code  | string   | code of SKU  |
| customer_code  | string  | code of customer   |
| qty  | float  | demand quantity  |


## Usage:
```python
import forecastability
fa = forecastability.Forecastability(data, tm="date")
# calculate frequency
fa.frequency()
# calculate stability
fa.stability()
# calculate periodicity
fa.periodicity()
# calculate single customer percent
fa.single_customer_percent()
# render forecastability report 
fa.render("forecastability_report.html")
```

## Reference:
[1] [时间周期序列周期性挖掘](https://wenku.baidu.com/view/8ad300afb8f67c1cfad6b87a.html) 

[2] [供应链三道防线：需求预测，库存计划，供应链执行](https://book.douban.com/subject/30223850/) 

[3] [Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.](https://otexts.com/fpp2/)
