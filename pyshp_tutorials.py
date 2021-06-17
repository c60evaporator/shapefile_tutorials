###### pyshpで読込＆shapelyとosgeoで各種操作 ######
# %%
import shapefile
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# 入力ファイルのパス
DAM_PATH = './W01-14_GML/W01-14-g_Dam.shp'  # 国交省ダムデータ（ポイント、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W01.html）
RIVER_PATH = './W05-09_25_GML/W05-09_25-g_Stream.shp'  # 国交省滋賀県河川データ（ライン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html）
LAKE_PATH = './W09-05_GML/W09-05-g_Lake.shp'  # 国交省湖沼データ（ポリゴン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W09-v2_2.html）
FARM_PATH= './01206釧路市（2020公開）/01206釧路市（2020公開）_5.shp'  # 農水省農場データ（ポリゴン、https://www.maff.go.jp/j/tokei/porigon/hudeporidl.html）

# shpファイル読込（pyshpライブラリ使用）
# ポイント（ダムデータ）
src_dam = shapefile.Reader(DAM_PATH, encoding='cp932')
shps_dam = src_dam.shapes()  # シェープ（位置情報）
recs_dam = src_dam.records()  # レコード（属性情報）
print(f'number of shapes = {len(shps_dam)}')  # シェープの数
print(f'number of records = {len(recs_dam)}')  # レコードの数
# ライン（河川データ）
src_river = shapefile.Reader(RIVER_PATH, encoding='cp932')
shps_river = src_river.shapes()
recs_river = src_river.records()
# ポリゴン（湖沼データ）
src_lake = shapefile.Reader(LAKE_PATH, encoding='cp932')
shps_lake = src_lake.shapes()
recs_lake = src_lake.records()
# ポリゴン（農場データ）
src_farm = shapefile.Reader(FARM_PATH, encoding='cp932')
shps_farm = src_farm.shapes()
recs_farm = src_farm.records()

# %% 共通操作1：位置情報と属性情報を取得
# # ポイント
# for shp, rec in zip(shps_dam, recs_dam):
#     print(f'ポイント位置{shp.points[0]}')  # 位置情報
#     print(f'ダム名:{rec["W01_001"]}\n堤高:{rec["W01_007"]}m\n総貯水量:{rec["W01_010"]}千m3\n所在地:{rec["W01_013"]}')  # 属性情報

# # ライン
# for shp, rec in zip(shps_river, recs_river):
#     print(f'ライン位置{shp.points}')  # 位置情報
#     print(f'河川名:{rec["W05_004"]}')  # 属性情報

# # ポリゴン
# for shp, rec in zip(shps_lake, recs_lake):
#     print(f'ポリゴン位置{shp.points}')  # 位置情報
#     print(f'湖沼名:{rec["W09_001"]}\n最大水深:{rec["W09_003"]}千m3\n水面標高:{rec["W09_004"]}')  # 属性情報

# %% 共通操作3：座標変換（osgeo.ogrライブラリ使用）
from osgeo import ogr, osr
# 座標系取得（osgeo.ogrライブラリ使用）
# # shpファイルから座標系を取得（https://tm23forest.com/contents/python-gdal-ogr-coordinatetransformation）
# shp = ogr.GetDriverByName('ESRI Shapefile').Open(FARM_PATH, 0)
# src_srs = shp.GetLayer().GetSpatialRef()
# src_srs_name = src_srs.GetName() if src_srs is not None else 'なし'
# print(f'変換前の座標系{src_srs_name}')
# # 変換後の座標系を指定
# dst_srs = osr.SpatialReference()
# dst_srs.ImportFromEPSG(4612)  # EPSGコードを指定（https://tmizu23.hatenablog.com/entry/20091215/1260868350）
# print(f'変換後の座標系{dst_srs.GetName()}')
# # 変換式を作成（参考https://gdal.org/python/osgeo.osr.CoordinateTransformation-class.html）
# trans = osr.CoordinateTransformation(src_srs, dst_srs)
# shp.Destroy()  # shapefileを閉じる

# # 取得した座標系を基に座標変換（pyshpライブラリで読み込んだデータにosgeoライブラリで作成した変換式適用）
# for shp in shps_farm:
#     print(f'変換前ポリゴン位置{shp.points}')  # 位置情報（座標変換前）
#     # 座標変換を実行
#     points_transfer = list(map(lambda point: trans.TransformPoint(point[0], point[1])[:2], shp.points))
#     print(f'変換後ポリゴン位置{points_transfer}')  # 位置情報（座標変換後）

# %% 共通操作3：座標変換（pyprojライブラリ使用）
# from pyproj import Transformer
# # 座標系取得（pyprojライブラリ使用）
# # 変換前の座標系を指定
# src_proj = "EPSG:2455"
# # 変換後の座標系を指定
# dst_proj = "EPSG:4612"
# # 変換式を作成
# transformer = Transformer.from_crs(src_proj, dst_proj)

# # 取得した座標系を基に座標変換（pyshpライブラリで読み込んだデータにosgeoライブラリで作成した変換式適用）
# for shp in shps_farm:
#     print(f'変換前ポリゴン位置{shp.points}')  # 位置情報（座標変換前）
#     # 座標変換を実行
#     points_transfer = list(map(lambda point: transformer.transform(point[0], point[1]), shp.points))
#     print(f'変換後ポリゴン位置{points_transfer}')  # 位置情報（座標変換後）


# # %% ポイントデータ操作1: ポイントデータ変換（shapelyライブラリ使用）
# for shp in shps_dam:
#     points = Point(shp.points[0])
#     #print(f'経度{points.x}, 緯度{points.y}')  # 位置情報

# # %% ポイントデータ操作2: ポイント間の距離を測定（shapelyライブラリ使用）
# point1 = Point(0, 0)
# point2 = Point(1, 1)
# dist = point1.distance(point2)
# print(f'距離={dist}')

# # %% ポイントデータ操作3: 緯度経度からポイント間の距離を測定（pyprojで平面座標に変換＋shapelyライブラリ使用）
# # 県庁所在地の座標を読込
# df_prefecture = pd.read_csv('./prefecture_location.csv', encoding='cp932')
# # 緯度経度の分秒を小数に変換

# df_prefecture['longitude'] = df_prefecture['都道府県庁 経度'].apply(lambda x: float(x.split('°')[0])
#                                                                    + float(x.split('°')[1].split("'")[0]) / 60
#                                                                    + float(x.split("'")[1].split('"')[0]) / 3600)
# df_prefecture['latitude'] = df_prefecture['都道府県庁 緯度'].apply(lambda x: float(x.split('°')[0])
#                                                                   + float(x.split('°')[1].split("'")[0]) / 60
#                                                                   + float(x.split("'")[1].split('"')[0]) / 3600)
# # 県庁所在地の緯度経度をDictionary化
# dict_pref_office = {row['都道府県']: (row['longitude'], row['latitude']) for i, row in df_prefecture.iterrows()}

# # UTM座標（ゾーン54=EPSG3100）への座標変換式を作成
# from pyproj import Transformer
# transformer = Transformer.from_crs('EPSG:4612', 'EPSG:3100')

# # ダムデータを1点ずつ走査
# for shp, rec in zip(shps_dam, recs_dam):
#     # ダムの位置（平面直角座標に変換）
#     dam_point = transformer.transform(shp.points[0][1], shp.points[0][0])
#     # 都道府県名を所在地から正規表現で抜き出して位置変換
#     prefecture = re.match('..*?県|..*?府|東京都|北海道', rec["W01_013"]).group()
#     pref_office_point = dict_pref_office[prefecture]  # 県庁所在地の緯度経度
#     pref_office_point = transformer.transform(pref_office_point[1], pref_office_point[0])
#     # ポイントデータとして格納（shapelyライブラリ使用）
#     dam_point = Point(dam_point)
#     pref_office_point = Point(pref_office_point)
#     # 距離を計算（shapelyライブラリ使用）
#     dist = dam_point.distance(pref_office_point)
#     print(f'{rec["W01_001"]}ダム {prefecture}庁まで{dist/1000}km')

# # %% ポイントデータ操作3: 緯度経度からポイント間の距離を測定（pyproj.Geod使用https://ikatakos.com/pot/programming/python/packages/pyproj）
# import pyproj
# # 距離測定用のGRS80楕円体
# grs80 = pyproj.Geod(ellps='GRS80')

# # ダムデータを1点ずつ走査
# for shp, rec in zip(shps_dam, recs_dam):
#     # ダムの位置（緯度経度）
#     dam_point = shp.points[0]
#     # 都道府県名を所在地から正規表現で抜き出し
#     prefecture = re.match('..*?県|..*?府|東京都|北海道', rec["W01_013"]).group()
#     pref_office_point = dict_pref_office[prefecture]  # 県庁所在地の緯度経度
#     # 距離を計算（pyproj.Geod使用）
#     azimuth, bkw_azimuth, dist = grs80.inv(dam_point[0], dam_point[1], pref_office_point[0], pref_office_point[1])
#     print(f'{rec["W01_001"]}ダム {prefecture}庁まで{dist/1000}km')

# %% ポイントデータ操作4: 最も近い点を探す
from sklearn.neighbors import NearestNeighbors
# UTM座標（ゾーン53=EPSG3099）への座標変換式を作成
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# 全点の位置関係を学習
dam_points_utm = [trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2] for shp in shps_dam] # UTM座標に変換
dam_points_array = np.array(dam_points_utm)  # ndarray化
nn = NearestNeighbors(algorithm='ball_tree')
nn.fit(dam_points_array)

# ダムデータを1点ずつ走査
for shp, rec in zip(shps_dam, recs_dam):
    # 最近傍点を探索
    point_utm = trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2]  # UTM座標に変換
    point_utm = np.array([list(point_utm)])  # ndarrayに変換
    dists, result = nn.kneighbors(point_utm, n_neighbors=3)  # 近傍上位3点を探索
    # 見付かった最近傍点(1番目は自身なので、2番目に近い点が最近傍点)を表示
    rec_nearest = recs_dam[result[0][1]]  # 最近傍点の属性データ取得
    dist_nearest = dists[0][1]/1000  # 最近傍点までの距離(m単位なのでkm単位に変換ん)
    print(f'{rec["W01_001"]}ダムから最も近いダム: {rec_nearest["W01_001"]}ダム  距離={dist_nearest}km')


# %% ポイントデータの処理（一括）
# shpファイル読込（pyshpライブラリ使用）
src = shapefile.Reader(FARM_PATH, encoding='SHIFT-JIS')
shps = src.shapes()
# ポイントデータとして格納（shapelyライブラリ使用）
points = shps

# ポイントに最も近いポイント

# ポリゴンに含まれるポイント

# %% 座標変換
###### GDALで座標系取得 ######
# shpファイルから座標系を取得（https://tm23forest.com/contents/python-gdal-ogr-coordinatetransformation）
shp = ogr.GetDriverByName('ESRI Shapefile').Open(FARM_PATH, 0)
src_srs = shp.GetLayer().GetSpatialRef()
src_srs_name = src_srs.GetName() if src_srs is not None else 'なし'
print(f'変換前の座標系{src_srs_name}')
# 変換後の座標系を指定
dst_srs = osr.SpatialReference()
dst_srs.ImportFromEPSG(4612)  # EPSGコードを指定（https://tmizu23.hatenablog.com/entry/20091215/1260868350）
print(f'変換後の座標系{dst_srs.GetName()}')
# 変換式を作成（参考https://gdal.org/python/osgeo.osr.CoordinateTransformation-class.html）
trans = osr.CoordinateTransformation(src_srs, dst_srs)
trans_reverse = osr.CoordinateTransformation(dst_srs, src_srs)

# shpファイル読込（pyshpライブラリ使用）
src = shapefile.Reader(FARM_PATH, encoding='SHIFT-JIS')
shps = src.shapes()
# 変換前の平面直角座標系で重心算出（shapleyライブラリ使用）
xys = [shp.points for shp in shps]
xy_centroids = [list(list(Polygon(points).centroid.coords)[0]) for points in xys]
# ポリゴンを緯度経度座標に変換（osgeoライブラリ使用）
lls = [list(map(lambda point: trans.TransformPoint(point[0], point[1])[:2], points)) for points in xys]
# 緯度経度座標の重心算出（shapleyライブラリ使用）
ll_centroids = [list(list(Polygon(points).centroid.coords)[0]) for points in lls]

# 座標の確認
xy_centorids = np.array(xy_centroids)  # 表示用にndarrayに変換
ll_centroids = np.array(ll_centroids)  # 表示用にndarrayに変換
print(xy_centorids[:5,:])  # 上から5データの平面直角座標表示
print(ll_centroids[:5,:])  # 上から5データの緯度経度表示

# 変換前と変換後の座標比較
plt.scatter(xy_centorids[:,0], ll_centroids[:,1])
plt.xlabel('y')
plt.ylabel('longitude')
plt.show()
plt.scatter(xy_centorids[:,1], ll_centroids[:,0])
plt.xlabel('x')
plt.ylabel('latitude')
plt.show()
# %%
