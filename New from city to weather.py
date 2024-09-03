import pandas as pd
import requests
import os
from tqdm import tqdm
import datetime

# 定义保存进度的文件
progress_file_path = '/Users/shang/Desktop/IC/Sem3/Cudd/cuddpro.csv'
results_file_path = '/Users/shang/Desktop/IC/Sem3/Cudd/cuddre.csv'
failed_requests_file_path = '/Users/shang/Desktop/IC/Sem3/Cudd/cuddfail.csv'

# 读取数据文件
file_path = 'C:/Users/shang/Desktop/IC/Sem3/Cudd/Cudduse.csv'
data = pd.read_csv(file_path, delimiter=',', engine='python')

# 确保日期列格式正确
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')


# 提取 ZeroTimes 列
zerotimes = data['ZeroTimes']

# 定义获取天气数据的函数
def get_weather_data(city, date, api_key):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    params = {
        'unitGroup': 'metric',
        'key': api_key,
        'contentType': 'json'
    }
    response = requests.get(f"{base_url}/{city}/{date.strftime('%Y-%m-%d')}", params=params)
    if response.status_code == 200:
        data = response.json()
        if 'days' in data and len(data['days']) > 0:
            day_data = data['days'][0]
            weather = {
                'tempmax': day_data.get('tempmax', None),
                'tempmin': day_data.get('tempmin', None),
                'temp': day_data.get('temp', None),
                'feelslikemax': day_data.get('feelslikemax', None),
                'feelslikemin': day_data.get('feelslikemin', None),
                'feelslike': day_data.get('feelslike', None),
                'dew': day_data.get('dew', None),
                'humidity': day_data.get('humidity', None),
                'precip': day_data.get('precip', None),
                'precipprob': day_data.get('precipprob', None),
                'precipcover': day_data.get('precipcover', None),
                'preciptype': day_data.get('preciptype', None),
                'snow': day_data.get('snow', None),
                'snowdepth': day_data.get('snowdepth', None),
                'windgust': day_data.get('windgust', None),
                'windspeed': day_data.get('windspeed', None),
                'winddir': day_data.get('winddir', None),
                'pressure': day_data.get('pressure', None),
                'cloudcover': day_data.get('cloudcover', None),
                'visibility': day_data.get('visibility', None),
                'solarradiation': day_data.get('solarradiation', None),
                'solarenergy': day_data.get('solarenergy', None),
                'uvindex': day_data.get('uvindex', None),
                'severerisk': day_data.get('severerisk', None),
                'sunrise': day_data.get('sunrise', None),
                'sunset': day_data.get('sunset', None)
            }
            return weather, None
        else:
            return None, f"No weather data found for {city} on {date}"
    else:
        return None, f"Failed to get data for {city} on {date}: {response.status_code}"

# 您的API密钥
api_key = 'UBQMP3EZ5XYN8QCUSYYY9CQV5'


# 如果结果文件存在，则读取已保存的部分
if os.path.exists(results_file_path):
    results = pd.read_csv(results_file_path).to_dict('records')
else:
    results = []

# 如果失败请求文件存在，则读取已保存的部分
if os.path.exists(failed_requests_file_path):
    failed_requests = pd.read_csv(failed_requests_file_path).to_dict('records')
else:
    failed_requests = []

# 定义保存数据的函数
def save_partial_results(results, file_path):
    weather_df = pd.DataFrame(results)
    weather_df.to_csv(file_path, index=False)

# 定义保存失败请求日志的函数
def save_failed_requests(failed_requests, file_path):
    failed_df = pd.DataFrame(failed_requests, columns=['City', 'Date', 'Reason'])
    failed_df.to_csv(file_path, index=False)

# 定义保存进度的函数
def save_progress(index, progress_file_path):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    progress_df = pd.DataFrame([{'Index': index, 'Timestamp': timestamp}])
    progress_df.to_csv(progress_file_path, index=False)

# 遍历每个城市和日期，处理1000条数据后保存一次
batch_size = 1000
output_file_path = results_file_path
failed_requests_file_path = failed_requests_file_path

for index, row in tqdm(data.iterrows(), total=data.shape[0], initial=1):
    if index < 1:
        continue
    city = row['Location.name']
    date = row['Date']
    if pd.isna(date):  # 跳过日期为 NaT 的行
        continue
    weather_data, error_reason = get_weather_data(city, date, api_key)
    if weather_data:
        results.append({
            'City': city,
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature Max': weather_data['tempmax'],
            'Temperature Min': weather_data['tempmin'],
            'Temperature': weather_data['temp'],
            'Feels Like Max': weather_data['feelslikemax'],
            'Feels Like Min': weather_data['feelslikemin'],
            'Feels Like': weather_data['feelslike'],
            'Dew Point': weather_data['dew'],
            'Humidity': weather_data['humidity'],
            'Precipitation': weather_data['precip'],
            'Precipitation Probability': weather_data['precipprob'],
            'Precipitation Cover': weather_data['precipcover'],
            'Precipitation Type': weather_data['preciptype'],
            'Snow': weather_data['snow'],
            'Snow Depth': weather_data['snowdepth'],
            'Wind Gust': weather_data['windgust'],
            'Wind Speed': weather_data['windspeed'],
            'Wind Direction': weather_data['winddir'],
            'Pressure': weather_data['pressure'],
            'Cloud Cover': weather_data['cloudcover'],
            'Visibility': weather_data['visibility'],
            'Solar Radiation': weather_data['solarradiation'],
            'Solar Energy': weather_data['solarenergy'],
            'UV Index': weather_data['uvindex'],
            'Severe Risk': weather_data['severerisk'],
            'Sunrise': weather_data['sunrise'],
            'Sunset': weather_data['sunset']
        })
    else:
        results.append({
            'City': city,
            'Date': date.strftime('%Y-%m-%d'),
            'Temperature Max': None,
            'Temperature Min': None,
            'Temperature': None,
            'Feels Like Max': None,
            'Feels Like Min': None,
            'Feels Like': None,
            'Dew Point': None,
            'Humidity': None,
            'Precipitation': None,
            'Precipitation Probability': None,
            'Precipitation Cover': None,
            'Precipitation Type': None,
            'Snow': None,
            'Snow Depth': None,
            'Wind Gust': None,
            'Wind Speed': None,
            'Wind Direction': None,
            'Pressure': None,
            'Cloud Cover': None,
            'Visibility': None,
            'Solar Radiation': None,
            'Solar Energy': None,
            'UV Index': None,
            'Severe Risk': None,
            'Sunrise': None,
            'Sunset': None
        })
        failed_requests.append({
            'City': city,
            'Date': date.strftime('%Y-%m-%d'),
            'Reason': error_reason
        })
    # 每处理1000条数据保存一次
    if (index + 1) % batch_size == 0:
        save_partial_results(results, output_file_path)
        save_failed_requests(failed_requests, failed_requests_file_path)
        save_progress(index + 1, progress_file_path)

# 处理完剩余的数据后保存
if len(results) % batch_size != 0:
    save_partial_results(results, output_file_path)
    save_failed_requests(failed_requests, failed_requests_file_path)
    save_progress(index + 1, progress_file_path)

# 读取结果文件
results_df = pd.read_csv(results_file_path)

# 添加 ZeroTimes 列到结果数据框中
results_df['ZeroTimes'] = zerotimes.values

# 保存最终结果
final_results_file_path = '/Users/shang/Desktop/IC/Sem3/Cudd/Cuddwea.csv'
results_df.to_csv(final_results_file_path, index=False)

print("数据处理完成，最终结果已保存到:", final_results_file_path)
