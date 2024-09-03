import geopandas as gpd
import matplotlib.pyplot as plt

# 读取印度地图的 shapefile
india = gpd.read_file("C:/Users/shang/Downloads/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# 过滤出印度
india = india[india.NAME == "India"]

# 定义城市的名称、经纬度和文本偏移量
cities = {
    'Sitapur': {'lat': 27.5680, 'lon': 80.6790, 'offset_x': -4, 'offset_y': 0},
    'Bahraich': {'lat': 27.5705, 'lon': 81.5977, 'offset_x': 1, 'offset_y': 0},
    'Barabanki': {'lat': 26.9268, 'lon': 81.1834, 'offset_x': 1, 'offset_y': -0.3},
    'Chennai': {'lat': 13.0843, 'lon': 80.2705, 'offset_x': 1, 'offset_y': 0},
    'Belgaum': {'lat': 15.8497, 'lon': 74.4977, 'offset_x': 1, 'offset_y': 0.3},
    'Bengaluru Urban': {'lat': 12.9700, 'lon': 77.6536, 'offset_x': -7.5, 'offset_y': -0.3},
    'Cuddalore': {'lat': 11.7480, 'lon': 79.7714, 'offset_x': 1, 'offset_y': 0},
}

# 创建一个 matplotlib 图形
fig, ax = plt.subplots(figsize=(10, 10))

# 在图形上绘制印度地图
india.plot(ax=ax, color='lightgrey')

# 绘制城市的位置和名称，并应用偏移量
for city, info in cities.items():
    ax.plot(info['lon'], info['lat'], marker='o', color='red', markersize=5)
    ax.text(info['lon'] + info['offset_x'], info['lat'] + info['offset_y'], city, fontsize=12)

# 设置标题和显示地图
ax.set_title("Selected Cities in India")
plt.show()
