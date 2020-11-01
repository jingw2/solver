#-*-coding:utf-8-*-
from datetime import datetime
import pandas as pd 


def date_converter(x):
    '''
    转换为日期格式
    '''
    if x is None:
        return x
    try:
        x = str(x)
    except Exception:
        return x
    
    try:
        return datetime.strptime(x, "%Y-%m-%d")
    except Exception:
        try:
            return datetime.strptime(x, "%Y/%m/%d")
        except Exception:
            try:
                return datetime.strptime(x, "%Y%m%d")
            except Exception:
                return x


def date_parser(x):
    '''
    日期格式转换为string
    '''
    if not isinstance(x, datetime):
        return None
    
    try:
        return x.strftime("%Y-%m-%d")
    except Exception:
        try:
            return x.strptime("%Y/%m/%d")
        except Exception:
            try:
                return x.strptime("%Y%m%d")
            except Exception:
                return None


def fill_ts(data, tm):
    '''
    填充时间序列，只保留两列，[ts, y]
    '''
    data[tm] = data[tm].apply(date_parser)
    if tm == "date":
        min_dt = date_converter(data[tm].min())
        max_dt = date_converter(data[tm].max())
        tm_list = [date_parser(x) for x in pd.date_range(start=min_dt, end=max_dt)]
    else:
        min_dt = data[tm].min()
        max_dt = data[tm].max()
        tm_list = list(range(min_dt, max_dt+1))
    tm_df = pd.DataFrame(tm_list, columns=[tm])
    df = pd.merge(tm_df, data[[tm, "sku_code", "qty"]], on=tm, how="left")
    df["qty"].fillna(0, inplace=True)
    return df 


def get_table(headers, rows, tablename):
    table_style = '''
        <style>
            .fl-table {
                margin: 20px;
                border-radius: 5px;
                font-size: 12px;
                border: none;
                border-collapse: collapse;
                max-width: 100%;
                white-space: nowrap;
                word-break: keep-all;
                font: Microsoft Yahei;
            }

            .fl-table th {
                text-align: left;
                font-size: 20px;
                font-family: Microsoft Yahei;
            }

            .fl-table tr {
                display: table-row;
                vertical-align: inherit;
                border-color: inherit;
                font-family: Microsoft Yahei;
            }

            .fl-table tr:hover td {
                background: #00d1b2;
                color: #F8F8F8;
            }

            .fl-table td, .fl-table th {
                border-style: none;
                border-top: 1px solid #dbdbdb;
                border-left: 1px solid #dbdbdb;
                border-bottom: 3px solid #dbdbdb;
                border-right: 1px solid #dbdbdb;
                padding: .5em .55em;
                font-size: 15px;
                font-family: Microsoft Yahei;
            }

            .fl-table td {
                border-style: none;
                font-family: Microsoft Yahei;
                font-size: 15px;
                vertical-align: center;
                border-bottom: 1px solid #dbdbdb;
                border-left: 1px solid #dbdbdb;
                border-right: 1px solid #dbdbdb;
                height: 30px;
                
            }

            .fl-table tr:nth-child(even) {
                background: #F8F8F8;
            }
        </style>
        
    '''
    title = '<div class="chart-container" style=""><p class="title" style="font-size: 18px; font-weight:bold; font-family: Microsoft Yahei" ></p>{}<p class="subtitle" style="font-size: 12px; font-family: "Microsoft Yahei"" > </p><table class="fl-table">'.format(tablename)
    table = '<tr>'
    for h in headers:
        table += "<th> {} </th>".format(h)
    table += '</tr>'

    for r in rows:
        table += '<tr>'
        for ele in r:
            table += '<td>{}</td>'.format(ele)
        table += '</tr>'
    end = '</table></div>'

    return table_style + title + table + end 

def get_line_charts(x, y, title, xname, yname):
    params = {
                'title': {
                    'text': title,
                },
                'toolbox': {
                    'show': 'true',
                    'orient': 'vertical',
                    'left': 'right',
                    'top': 'center',
                    'feature': {
                        'mark': {'show': 'true'},
                        'dataView': {'show': 'true', 'readOnly': 'false'},
                        'magicType': {'show': 'true', 'type': ['line', 'bar', 'stack', 'tiled']},
                        'restore': {'show': 'true'},
                        'saveAsImage': {'show': 'true'}
                    }
                },
                'tooltip': {
                    "show": 'true',
                    'trigger': 'axis'
                },
                'legend': {
                    'data': []
                },
                'xAxis': {
                    'data': x,
                    'name': xname,
                    "nameLocation": "middle",
                    "nameGap": 25,
                    'nameTextStyle': {
                        'fontSize': 14
                    }
                },
                'yAxis': {
                    'name': yname,
                    'type': 'value',
                    "nameLocation": "middle",
                    "nameGap": 40,
                    'nameTextStyle': {
                        'fontSize': 14
                    },
                    "axisLabel": {
                        "show": 'true',
                        "position": "right",
                        "margin": 8,
                        "formatter": "{value}%"
                    }
                },
                'series': [{
                    'name': "",
                    'type': 'line',
                    'data': y
                }]
            }
    chart = ('''
        <div id="main" style="width: 800px;height:500px;"></div>
        <script type="text/javascript">
            // based on prepared DOM, initialize echarts instance
            var myChart = echarts.init(document.getElementById('main'));

            // specify chart configuration item and data
            var option = %s;

            // use configuration item and data specified to show chart
            myChart.setOption(option);
        </script>
    ''' % str(params))
    return chart 
