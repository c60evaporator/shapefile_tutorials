# %% 読込1：Shapefileの読込
import geopandas as gpd
from shapely.geometry import Point
import pyproj
import pandas as pd
import numpy as np
import re

# 入力ファイルのパス
DAM_PATH = './W01-14_GML/W01-14-g_Dam.shp'  # 国交省ダムデータ（ポイント、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W01.html）
RIVER_PATH = './W05-09_25_GML/W05-09_25-g_Stream.shp'  # 国交省滋賀県河川データ（ライン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html）
LAKE_PATH = './W09-05_GML/W09-05-g_Lake.shp'  # 国交省湖沼データ（ポリゴン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W09-v2_2.html）
FARM_PATH= './01206釧路市（2020公開）/01206釧路市（2020公開）_5.shp'  # 農水省農場データ（ポリゴン、https://www.maff.go.jp/j/tokei/porigon/hudeporidl.html）

# shpファイル読込
# ポイント（ダムデータ）
gdf_dam = gpd.read_file(DAM_PATH, encoding='cp932')  # Shapefile読込
print(f'number of records = {len(gdf_dam)}')  # レコード数
# ライン（河川データ）
gdf_river = gpd.read_file(RIVER_PATH, encoding='cp932')  # Shapefile読込
# ポリゴン（湖沼データ）
gdf_lake = gpd.read_file(LAKE_PATH, encoding='cp932')  # Shapefile読込
# ポリゴン（農場データ）
gdf_farm = gpd.read_file(FARM_PATH, encoding='cp932')  # Shapefile読込

# %% 読込2：ジオメトリ情報と属性情報を1レコードずつ取得
# ポイント
for i, row in gdf_dam.iterrows():
    print(f'ポイント位置{row["geometry"]}')  # ジオメトリ情報
    print(f'ダム名:{row["W01_001"]}\n堤高:{row["W01_007"]}m\n総貯水量:{row["W01_010"]}千m3\n所在地:{row["W01_013"]}')  # 属性情報

# ライン
for i, row in gdf_river.iterrows():
    print(f'ライン位置{row["geometry"]}')  # ジオメトリ情報
    print(f'河川名:{row["W05_004"]}')  # 属性情報

# ポリゴン
for i, row in gdf_lake.iterrows():
    print(f'ポリゴン位置{row["geometry"]}')  # ジオメトリ情報
    print(f'湖沼名:{row["W09_001"]}\n最大水深:{row["W09_003"]}千m3\n水面標高:{row["W09_004"]}')  # 属性情報

# %% 読込3：GeoJSON読込

# %% 処理1：一括座標変換（変換前座標を取得できるとき）
# 変換後の座標系指定（平面直角座標13系(EPSG2455) → 緯度経度(EPSG4612)）
dst_proj = 4612 # 変換後の座標系を指定

# 座標変換
gdf_farm_transfer = gdf_farm.to_crs(epsg=dst_proj)  # 変換式を作成

for (i1, row1), (i2, row2) in zip(gdf_farm.iterrows(), gdf_farm_transfer.iterrows()):
    print(f'変換前ポリゴン位置{row1["geometry"]}')  # 位置情報（座標変換前）
    print(f'変換後ポリゴン位置{row2["geometry"]}')  # 位置情報（座標変換後）

# %% 処理1 一括座標変換（変換前座標を取得できないとき）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
src_proj = 4612  # 変換前の座標系を指定
dst_proj = 3099  # 変換後の座標系を指定

# ポイント（ダムデータ、TransformPointの引数は緯度,経度の順番で指定）
gdf_dam_utm = gdf_dam.copy()
gdf_dam_utm.crs = f'epsg:{src_proj}'  # 変換前座標を指定
gdf_dam_utm = gdf_dam_utm.to_crs(epsg=dst_proj)  # 変換後座標に変換

# ライン（河川データ）
gdf_river_utm = gdf_river.copy()
gdf_river_utm.crs = f'epsg:{src_proj}'  # 変換前座標を指定
gdf_river_utm = gdf_river_utm.to_crs(epsg=dst_proj)  # 変換後座標に変換

# ポリゴン（湖沼データ）
gdf_lake_utm = gdf_lake.copy()
gdf_lake_utm.crs = f'epsg:{src_proj}'  # 変換前座標を指定
gdf_lake_utm = gdf_lake_utm.to_crs(epsg=dst_proj)  # 変換後座標に変換

# %% 処理3(ポイントデータ操作2): ポイント間の距離を測定（平面座標系）
# 2点が格納されたGeoDataFrame作成
gdf_new = gpd.GeoDataFrame(crs = 'epsg:3099')  # UTM座標53N系(EPSG3099)のGeoDataFrameを作成
gdf_new['geometry'] = None
gdf_new.at[0, 'geometry'] = Point(0, 0)
gdf_new.at[1, 'geometry'] = Point(1, 1)
print(gdf_new)
# 距離測定
dist = gdf_new.at[0, 'geometry'].distance(gdf_new.at[1, 'geometry'])  # 0行目と1行目の距離を測定
print(f'距離={dist}')

# %% 処理3用のデータ（都道府県庁緯度経度のDict）準備
# 県庁所在地の座標を読込
df_prefecture = pd.read_csv('./prefecture_location.csv', encoding='cp932')
# 緯度経度の分秒を小数に変換

df_prefecture['longitude'] = df_prefecture['都道府県庁 経度'].apply(lambda x: float(x.split('°')[0])
                                                                   + float(x.split('°')[1].split("'")[0]) / 60
                                                                   + float(x.split("'")[1].split('"')[0]) / 3600)
df_prefecture['latitude'] = df_prefecture['都道府県庁 緯度'].apply(lambda x: float(x.split('°')[0])
                                                                  + float(x.split('°')[1].split("'")[0]) / 60
                                                                  + float(x.split("'")[1].split('"')[0]) / 3600)
# 県庁所在地の緯度経度をDictionary化
dict_pref_office = {row['都道府県']: (row['longitude'], row['latitude']) for i, row in df_prefecture.iterrows()}

# %% 処理3: 緯度経度からポイント間の距離を測定（pyproj.Geod使用）
# 距離測定用のGRS80楕円体
grs80 = pyproj.Geod(ellps='GRS80')

# 都道府県と都道府県庁の位置を紐づけ
gdf_dam['prefecture'] = gdf_dam['W01_013'].apply(
    lambda x: re.match('..*?県|..*?府|東京都|北海道', x).group())
gdf_dam['pref_office_point'] = gdf_dam['prefecture'].apply(
    lambda x: Point(*dict_pref_office[x]))

# ダムデータを1点ずつ走査
for i, row in gdf_dam.iterrows():
    # 距離を計算（pyprojライブラリ使用）
    azimuth, bkw_azimuth, dist = grs80.inv(row['geometry'].x, row['geometry'].y,
                                           row['pref_office_point'].x, row['pref_office_point'].y)
    print(f'{row["W01_001"]}ダム {row["prefecture"]}庁まで{dist/1000}km')

# %% 処理3: 緯度経度からポイント間の距離を一括測定（pyproj.Geod使用）
# 距離測定用のGRS80楕円体
grs80 = pyproj.Geod(ellps='GRS80')

# 都道府県と都道府県庁の位置を紐づけ
gdf_dam['prefecture'] = gdf_dam['W01_013'].apply(
    lambda x: re.match('..*?県|..*?府|東京都|北海道', x).group())
gdf_dam['pref_office_point'] = gdf_dam['prefecture'].apply(
    lambda x: Point(*dict_pref_office[x]))
# 距離を計算（pyproj.Geod使用）
gdf_dam['dist'] = gdf_dam.apply(
    lambda rec: grs80.inv(rec['geometry'].x, rec['geometry'].y,
                          rec['pref_office_point'].x, rec['pref_office_point'].y)[2], axis=1)

# 都道府県庁から最も遠いダムを表示
farthest_index = gdf_dam['dist'].idxmax(axis=1)
print(f'{gdf_dam.at[farthest_index, "prefecture"]}庁から最も遠いダム={gdf_dam.at[farthest_index, "W01_001"]}ダム  距離={gdf_dam.at[farthest_index, "dist"]/1000}km')

# %% 処理4: 最も近い点を探す
from sklearn.neighbors import NearestNeighbors
# 座標変換（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
gdf_dam_utm = gdf_dam.copy()
gdf_dam_utm.crs = f'epsg:{4612}'  # 変換前座標を指定
gdf_dam_utm = gdf_dam_utm.to_crs(epsg=3099)  # 変換後座標に変換

# 全点の位置関係を学習
dam_points_array = np.array([[point.x, point.y] for point in gdf_dam_utm['geometry']])  # ndarray化
nn = NearestNeighbors(algorithm='ball_tree')
nn.fit(dam_points_array)

# ダムデータを1点ずつ走査
for i, row in gdf_dam_utm.iterrows():
    # 最近傍点を探索
    point_utm = np.array([[row['geometry'].x, row['geometry'].y]])  # ndarrayに変換
    dists, result = nn.kneighbors(point_utm, n_neighbors=3)  # 近傍上位3点を探索
    # 見付かった最近傍点(1番目は自身なので、2番目に近い点が最近傍点)を表示
    row_nearest = gdf_dam_utm.loc[result[0][1]]  # 最近傍点のデータ取得
    dist_nearest = dists[0][1]/1000  # 最近傍点までの距離(m単位なのでkm単位に変換ん)
    print(f'{row["W01_001"]}ダムから最も近いダム: {row_nearest["W01_001"]}ダム  距離={dist_nearest}km')

# %% 処理4: 最も近い点を一括検索
from sklearn.neighbors import NearestNeighbors
# 座標変換（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
gdf_dam_utm = gdf_dam.copy()
gdf_dam_utm.crs = f'epsg:{4612}'  # 変換前座標を指定
gdf_dam_utm = gdf_dam_utm.to_crs(epsg=3099)  # 変換後座標に変換

# 全点の位置関係を学習
dam_points_array = np.array([[point.x, point.y] for point in gdf_dam_utm['geometry']])  # ndarray化
nn = NearestNeighbors(algorithm='ball_tree')
nn.fit(dam_points_array)

# 最近傍点を一括探索
dists, results = nn.kneighbors(dam_points_array)
# 見付かった最近傍点(1番目は自身なので、2番目に近い点が最近傍点)と距離を保持
dist_nearest = [dist/1000 for dist in dists[:,1]]  # 最近傍点までの距離(m単位なのでkm単位に変換)

# 最も近いダムからの距離が最大のダムを表示
farthest_index = np.argmax(dist_nearest)
print(f'最近傍ダムからの距離が最大のダム={gdf_dam_utm.at[farthest_index, "W01_001"]}ダム\
    最近傍ダム={gdf_dam_utm.at[results[farthest_index][1], "W01_001"]}ダム\
    距離={dist_nearest[farthest_index]}km')

# %% 処理6: ラインの長さ測定
# 座標変換（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
gdf_river_utm = gdf_river.copy()
gdf_river_utm.crs = f'epsg:{4612}'  # 変換前座標を指定
gdf_river_utm = gdf_river_utm.to_crs(epsg=3099)  # 変換後座標に変換

# 河川データを1点ずつ走査
for i, row in gdf_river_utm.iterrows():
    # 長さを算出して表示
    length = row.length
    print(f'{row["W05_004"]}  長さ={length}m')

# %% 処理6: 長さを一括測定
# 座標変換（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
gdf_river_utm = gdf_river.copy()
gdf_river_utm.crs = f'epsg:{4612}'  # 変換前座標を指定
gdf_river_utm = gdf_river_utm.to_crs(epsg=3099)  # 変換後座標に変換

# 長さを一括算出(km単位にするため1000で割る)
gdf_river_utm['river_length'] = gdf_river_utm.length / 1000

# 河川名グルーピングして長さ降順でソート
df_length = gdf_river_utm[['W05_004', 'river_length']].groupby('W05_004').sum(
    ).sort_values('river_length', ascending=False)
print(df_length.head(5))
# 名称不明と琵琶湖を除外
df_length = df_length[~df_length.index.isin(['名称不明', '琵琶湖'])]
print(df_length.head(5))

# %% 処理8: ポリゴンの重心測定（緯度経度座標のまま計算）
# 湖沼データを1点ずつ走査
for i, row in gdf_lake.iterrows():
    # 重心を算出
    center = list(row['geometry'].centroid.coords)[0]
    print(f'{row["W09_001"]}  重心={center}')

# %% 処理8: ポリゴンの重心測定（UTM座標に変換）
# 座標変換（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)）
gdf_lake_utm = gdf_lake.copy()
gdf_lake_utm.crs = f'epsg:{4612}'  # 変換前座標を指定
gdf_lake_utm = gdf_lake_utm.to_crs(epsg=3099)  # 変換後座標に変換

# 湖沼データを1点ずつ走査
for i, row in gdf_lake_utm.iterrows():
    # 重心を算出
    center_utm = row['geometry'].centroid
    # 緯度経度座標に戻す（pyprojを使用。TransformPointは緯度→経度の順で返すので、元の座標系に合わせ経度を先に反転させる）
    transformer = pyproj.Transformer.from_crs("EPSG:3099", "EPSG:4612")
    center = transformer.transform(center_utm.x, center_utm.y)[1::-1]
    print(f'{row["W09_001"]}  重心={center}')

    # 座標変換しなかった場合の重心との距離を計算
    center_not_trans = gdf_lake.at[i, 'geometry'].centroid
    grs80 = pyproj.Geod(ellps='GRS80')
    dist = grs80.inv(center[0], center[1], center_not_trans.x, center_not_trans.y)[2]
    print(f'{row["W09_001"]}  座標変換なしとの差={dist}m')

# %% 処理8: ポリゴン重心位置を一括取得
# 重心位置を一括計算
centers = gdf_lake['geometry'].centroid
# 重心が最も北にある湖を表示
northest_index = np.argmax([center.y for center in centers])
print(f'重心が最も北にある湖={gdf_lake.at[northest_index, "W09_001"]}  北緯{centers.at[northest_index].y}度')

# %%
