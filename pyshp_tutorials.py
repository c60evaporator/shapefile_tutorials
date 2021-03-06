# %% 読込1：Shapefileの読込
import shapefile
from shapely.geometry import Point, LineString, Polygon
from osgeo import ogr, osr
import pyproj
import numpy as np
import pandas as pd
import re

# 入力ファイルのパス
DAM_PATH = './W01-14_GML/W01-14-g_Dam.shp'  # 国交省ダムデータ（ポイント、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W01.html）
RIVER_PATH = './W05-09_25_GML/W05-09_25-g_Stream.shp'  # 国交省滋賀県河川データ（ライン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W05.html）
LAKE_PATH = './W09-05_GML/W09-05-g_Lake.shp'  # 国交省湖沼データ（ポリゴン、https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-W09-v2_2.html）
FARM_PATH= './01206釧路市（2020公開）/01206釧路市（2020公開）_5.shp'  # 農水省農場データ（ポリゴン、https://www.maff.go.jp/j/tokei/porigon/hudeporidl.html）

# shpファイル読込（pyshpライブラリ使用）
# ポイント（ダムデータ）
src_dam = shapefile.Reader(DAM_PATH, encoding='cp932')  # Shapefile読込
shps_dam = src_dam.shapes()  # シェープ（ジオメトリ情報）
recs_dam = src_dam.records()  # レコード（属性情報）
print(f'number of shapes = {len(shps_dam)}')  # シェープの数
print(f'number of records = {len(recs_dam)}')  # レコードの数
# ライン（河川データ）
src_river = shapefile.Reader(RIVER_PATH, encoding='cp932')  # Shapefile読込
shps_river = src_river.shapes()  # ジオメトリ情報
recs_river = src_river.records()  # 属性情報
# ポリゴン（湖沼データ）
src_lake = shapefile.Reader(LAKE_PATH, encoding='cp932')  # Shapefile読込
shps_lake = src_lake.shapes()  # ジオメトリ情報
recs_lake = src_lake.records()  # 属性情報
# ポリゴン（農場データ）
src_farm = shapefile.Reader(FARM_PATH, encoding='cp932')  # Shapefile読込
shps_farm = src_farm.shapes()  # ジオメトリ情報
recs_farm = src_farm.records()  # 属性情報

# %% 読込2：ジオメトリ情報と属性情報を1レコードずつ取得（pyshpライブラリ使用）
# ポイント
for shp, rec in zip(shps_dam, recs_dam):
    print(f'ポイント位置{shp.points[0]}')  # ジオメトリ情報
    print(f'ダム名:{rec["W01_001"]}\n堤高:{rec["W01_007"]}m\n総貯水量:{rec["W01_010"]}千m3\n所在地:{rec["W01_013"]}')  # 属性情報

# ライン
for shp, rec in zip(shps_river, recs_river):
    print(f'ライン位置{shp.points}')  # ジオメトリ情報
    print(f'河川名:{rec["W05_004"]}')  # 属性情報

# ポリゴン
for shp, rec in zip(shps_lake, recs_lake):
    print(f'ポリゴン位置{shp.points}')  # ジオメトリ情報
    print(f'湖沼名:{rec["W09_001"]}\n最大水深:{rec["W09_003"]}千m3\n水面標高:{rec["W09_004"]}')  # 属性情報

# %% 処理1：座標変換（osgeo.osrライブラリ使用）
# shpファイルから座標系取得（osgeo.ogrライブラリ使用、https://tm23forest.com/contents/python-gdal-ogr-coordinatetransformation）
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
shp.Destroy()  # shapefileを閉じる

# 取得した座標系を基に座標変換（pyshpライブラリで読み込んだデータにosgeoライブラリで作成した変換式適用）
for shp in shps_farm:
    print(f'変換前ポリゴン位置{shp.points}')  # 位置情報（座標変換前）
    # 座標変換を実行
    points_transfer = list(map(lambda point: trans.TransformPoint(point[0], point[1])[:2], shp.points))
    print(f'変換後ポリゴン位置{points_transfer}')  # 位置情報（座標変換後）

# %% 処理1：座標変換（pyprojライブラリ使用）
# 変換前後の座標系指定（平面直角座標13系(EPSG2455) → 緯度経度(EPSG4612)）
src_proj = 'EPSG:2455' # 変換前の座標系を指定
dst_proj = 'EPSG:4612' # 変換後の座標系を指定
transformer = pyproj.Transformer.from_crs(src_proj, dst_proj) # 変換式を作成

# 取得した座標系を基に座標変換（pyshpライブラリで読み込んだデータにpyprojライブラリで作成した変換式適用）
for shp in shps_farm:
    print(f'変換前ポリゴン位置{shp.points}')  # 位置情報（座標変換前）
    # 座標変換を実行
    points_transfer = list(map(lambda point: transformer.transform(point[0], point[1]), shp.points))
    print(f'変換後ポリゴン位置{points_transfer}')  # 位置情報（座標変換後）

# %% 処理1 一括座標変換（osgeo.osrライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# ポイント（ダムデータ、TransformPointの引数は緯度,経度の順番で指定）
dam_points_utm = [list(trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2]) for shp in shps_dam]

# ライン（河川データ）
river_lines_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_river]

# ポリゴン（湖沼データ）
lake_polys_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_lake]

# %% 処理2(ポイントデータ操作1): ポイントデータ変換（shapelyライブラリ使用）
for shp in shps_dam:
    points = Point(shp.points[0])
    print(f'経度{points.x}, 緯度{points.y}')  # 位置情報

# %% 処理3(ポイントデータ操作2): ポイント間の距離を測定（shapelyライブラリ使用）
point1 = Point(0, 0)
point2 = Point(1, 1)
dist = point1.distance(point2)
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

# %% 処理3(ポイントデータ操作2): 緯度経度からポイント間の距離を測定（osgeo.osrライブラリでUTM座標に変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標44N系(EPSG3100)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3100)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# ダムデータを1点ずつ走査
for shp, rec in zip(shps_dam, recs_dam):
    # ダムの位置（UTM座標に変換）
    dam_point = trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2]
    # 都道府県名を所在地から正規表現で抜き出して位置変換
    prefecture = re.match('..*?県|..*?府|東京都|北海道', rec['W01_013']).group()
    pref_office_point = dict_pref_office[prefecture]  # 県庁所在地の緯度経度
    pref_office_point = trans.TransformPoint(pref_office_point[1], pref_office_point[0])[:2]
    # ポイントデータとして格納（shapelyライブラリ使用）
    dam_point = Point(dam_point)
    pref_office_point = Point(pref_office_point)
    # 距離を計算（shapelyライブラリ使用）
    dist = dam_point.distance(pref_office_point)
    print(f'{rec["W01_001"]}ダム {prefecture}庁まで{dist/1000}km')

# %% 処理3(ポイントデータ操作2): 緯度経度からポイント間の距離を測定（pyproj.Geod使用https://ikatakos.com/pot/programming/python/packages/pyproj）
# 距離測定用のGRS80楕円体
grs80 = pyproj.Geod(ellps='GRS80')

# ダムデータを1点ずつ走査
for shp, rec in zip(shps_dam, recs_dam):
    # ダムの位置（緯度経度）
    dam_point = shp.points[0]
    # 都道府県名を所在地から正規表現で抜き出し
    prefecture = re.match('..*?県|..*?府|東京都|北海道', rec['W01_013']).group()
    pref_office_point = dict_pref_office[prefecture]  # 県庁所在地の緯度経度
    # 距離を計算（pyproj.Geod使用）
    azimuth, bkw_azimuth, dist = grs80.inv(dam_point[0], dam_point[1], pref_office_point[0], pref_office_point[1])
    print(f'{rec["W01_001"]}ダム {prefecture}庁まで{dist/1000}km')

# %% 処理3(ポイントデータ操作2): 緯度経度からポイント間の距離を一括測定（pyproj.Geod使用）
# 距離測定用のGRS80楕円体
grs80 = pyproj.Geod(ellps='GRS80')

# 都道府県名を所在地から正規表現で抜き出し
prefectures = [re.match('..*?県|..*?府|東京都|北海道', rec['W01_013']).group() for rec in recs_dam]
pref_office_points = [dict_pref_office[prefecture] for prefecture in prefectures]  # 県庁所在地の緯度経度
# 距離を計算（pyproj.Geod使用）
dists = [grs80.inv(shp.points[0][0], shp.points[0][1], pref[0], pref[1])[2] for shp, pref in zip(shps_dam, pref_office_points)]

# 都道府県庁から最も遠いダムを表示
farthest_index = np.argmax(dists)
print(f'{prefectures[farthest_index]}庁から最も遠いダム={recs_dam[farthest_index]["W01_001"]}ダム  距離={dists[farthest_index]/1000}km')

# %% 処理4(ポイントデータ操作3): 最も近い点を探す（osgeo.osrライブラリでUTM座標に変換＋scikit-learn使用）
from sklearn.neighbors import NearestNeighbors
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　Osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# UTM座標に変換
dam_points_utm = [trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2] for shp in shps_dam] 
# 全点の位置関係を学習
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

# %% 処理4(ポイントデータ操作3): 最も近い点を一括検索（osgeo.osrライブラリでUTM座標に変換＋scikit-learn使用）
from sklearn.neighbors import NearestNeighbors
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# UTM座標に変換
dam_points_utm = [trans.TransformPoint(shp.points[0][1], shp.points[0][0])[:2] for shp in shps_dam] 
# 全点の位置関係を学習
dam_points_array = np.array(dam_points_utm)  # ndarray化
nn = NearestNeighbors(algorithm='ball_tree')
nn.fit(dam_points_array)

# 最近傍点を一括探索
dists, results = nn.kneighbors(dam_points_array)
# 見付かった最近傍点(1番目は自身なので、2番目に近い点が最近傍点)と距離を保持
rec_nearest = [recs_dam[result] for result in results[:,1]]  # 最近傍点の属性データ取得
dist_nearest = [dist/1000 for dist in dists[:,1]]  # 最近傍点までの距離(m単位なのでkm単位に変換)

# 最も近いダムからの距離が最大のダムを表示
farthest_index = np.argmax(dist_nearest)
print(f'最近傍ダムからの距離が最大のダム={recs_dam[farthest_index]["W01_001"]}ダム\
    最近傍ダム={rec_nearest[farthest_index]["W01_001"]}ダム\
    距離={dist_nearest[farthest_index]}km')

# %% 処理5(ラインデータ操作1): ラインデータ変換（shapelyライブラリ使用）
for shp, rec in zip(shps_river, recs_river):
    line = LineString(shp.points)
    print(f'{rec["W05_004"]} {list(line.coords)}')  # 河川名と位置情報を表示

# %% 処理6(ラインデータ操作2): ラインの長さ測定（osgeo.osrライブラリでUTM座標に変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)

# 河川データを1点ずつ走査
for shp, rec in zip(shps_river, recs_river):
    # UTM座標に変換
    river_utm = list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points))
    # ラインデータに変換
    line = LineString(river_utm)
    # 長さを算出して表示
    length = line.length
    print(f'{rec["W05_004"]}  長さ={length}m')

# %% 処理6(ラインデータ操作2): 長さを一括測定（osgeo.osrライブラリでUTM座標に変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)
# UTM座標に一括変換
lines_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_river]
# ラインデータに変換し長さを一括算出(km単位にするため1000で割る)
length_list = [LineString(points).length/1000 for points in lines_utm]
# 河川名をリスト化
river_names = [rec['W05_004'] for rec in recs_river]

# 河川名の重複があるので、Pandas DataFrameに読み込んでグルーピング
df_river = pd.DataFrame(np.array([river_names, length_list]).T, columns=['name', 'length'])
df_river = df_river.astype({'length': float})
# 河川名グルーピングして長さ降順でソート
df_length = df_river.groupby('name').sum().sort_values('length', ascending=False)
print(df_length.head(5))
# 名称不明と琵琶湖を除外
df_length = df_length[~df_length.index.isin(['名称不明', '琵琶湖'])]
print(df_length.head(5))

# %% 処理7(ポリゴンデータ操作1): ポリゴンデータ変換（shapelyライブラリ使用）
for shp, rec in zip(shps_lake, recs_lake):
    # shp.partsに基づきポリゴン点を分割（穴を定義）
    parts = [i for i in shp.parts] + [len(shp.points) - 1]
    points_hole = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # ポリゴンを作成
    poly = Polygon(points_hole[0], points_hole[1:])
    print(f'{rec["W09_001"]} {list(poly.exterior.coords)}')  # 位置情報を表示

# %% 処理8(ポリゴンデータ操作2): ポリゴンの重心測定（shapelyライブラリ使用、緯度経度座標のまま計算）
# 湖沼データを1点ずつ走査
for shp, rec in zip(shps_lake, recs_lake):
    # shp.partsに基づきポリゴン点を分割（穴を定義）
    parts = [i for i in shp.parts] + [len(shp.points) - 1]
    points_hole = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # ポリゴンデータに変換
    poly = Polygon(points_hole[0], points_hole[1:])
    # 重心を算出
    center = list(poly.centroid.coords)[0]
    print(f'{rec["W09_001"]}  重心={center}')

# %% 処理8(ポリゴンデータ操作2): ポリゴンの重心測定（osgeo.osrライブラリでUTM座標に変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)  # 緯度経度→UTM座標への変換式
trans_reverse = osr.CoordinateTransformation(dst_srs, src_srs)  # UTM座標→緯度経度への逆変換

# 湖沼データを1点ずつ走査
for shp, rec in zip(shps_lake, recs_lake):
    # UTM座標に変換
    lake_utm = list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points))
    # shp.partsに基づきポリゴン点を分割（穴を定義）
    parts = [i for i in shp.parts] + [len(lake_utm) - 1]
    points_hole = [lake_utm[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # ポリゴンデータに変換
    poly = Polygon(points_hole[0], points_hole[1:])
    # 重心を算出
    center_utm = list(poly.centroid.coords)[0]
    # 緯度経度座標に戻す（TransformPointは緯度→経度の順で返すので、元の座標系に合わせ経度を先に反転させる）
    center = tuple(trans_reverse.TransformPoint(center_utm[0], center_utm[1])[1::-1])
    print(f'{rec["W09_001"]}  重心={center}')

    # 座標変換しなかった場合の重心との距離を計算
    parts_not_trans = [i for i in shp.parts] + [len(shp.points) - 1]
    points_hole_not_trans = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    center_not_trans = list(Polygon(points_hole_not_trans[0], points_hole_not_trans[1:]).centroid.coords)[0]
    grs80 = pyproj.Geod(ellps='GRS80')
    dist = grs80.inv(center[0], center[1], center_not_trans[0], center_not_trans[1])[2]
    print(f'{rec["W09_001"]}  座標変換なしとの差={dist}m')

# %% 処理8(ポリゴンデータ操作2): ポリゴン重心位置を一括取得（shapelyライブラリ使用）
# shp.partsに基づきポリゴン点を一括分割
parts_list = [[i for i in shp.parts] + [len(shp.points) - 1] for shp in shps_lake]
points_hole_list = [[shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)] for shp, parts in zip(shps_lake, parts_list)]
# 重心位置を一括計算
centers = [tuple(list(Polygon(points_hole[0], points_hole[1:]).centroid.coords)[0]) for points_hole in points_hole_list]
# 重心が最も北にある湖を表示
northest_index = np.argmax([center[1] for center in centers])
print(f'重心が最も北にある湖={recs_lake[northest_index]["W09_001"]}  北緯{centers[northest_index][1]}度')

# %% 処理9(ポリゴンデータ操作3): ポリゴンの面積測定（osgeo.osrライブラリでUTM座標に変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)  # 緯度経度→UTM座標への変換式

# 湖沼データを1点ずつ走査
for shp, rec in zip(shps_lake, recs_lake):
    # UTM座標に変換
    lake_utm = list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points))
    # shp.partsに基づきポリゴン点を分割（穴を定義）
    parts = [i for i in shp.parts] + [len(lake_utm) - 1]
    points_hole = [lake_utm[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # ポリゴンデータに変換
    poly = Polygon(points_hole[0], points_hole[1:])
    # 面積を算出（m2 → km2に単位変換）
    area = poly.area/1000000
    print(f'{rec["W09_001"]}  面積={area}km2')

# %% 処理9(ポリゴンデータ操作3): ポリゴン面積を一括取得（osgeo.osrライブラリでUTM座標に一括変換＋shapelyライブラリ使用）
# 変換前後の座標系指定（緯度経度(EPSG4612) → UTM座標53N系(EPSG3099)　osgeo.osrを使用）
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)  # 緯度経度→UTM座標への変換式
# UTM座標に一括変換
polys_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_lake]
# shp.partsに基づきポリゴン点を一括分割
parts_list = [[i for i in shp.parts] + [len(shp.points) - 1] for shp in shps_lake]
points_hole_list = [[poly[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)] for poly, parts in zip(polys_utm, parts_list)]
# 面積を一括計算
areas = [Polygon(points_hole[0], points_hole[1:]).area/1000000 for points_hole in points_hole_list]
# 面積最大の湖を表示
biggest_index = np.argmax(areas)
print(f'面積最大の湖={recs_lake[biggest_index]["W09_001"]}  面積={areas[biggest_index]}km2')

# %% 処理10: ジオコーディング（geopy + Nominatimを使用）
from geopy.geocoders import Nominatim
# 堤高150m以上のダムに絞る
over150m_indices = [i for i, rec in enumerate(recs_dam) if rec['W01_007'] > 150]
over150m_shps = [shp for i, shp in enumerate(shps_dam) if i in over150m_indices]
over150m_recs = [rec for i, rec in enumerate(recs_dam) if i in over150m_indices]

# Nominatimを指定
geolocator = Nominatim(user_agent='test')
# ダムを走査
for shp, rec in zip(over150m_shps, over150m_recs):
    # ジオコーディング
    name = f'{rec["W01_001"]}ダム'  # ダム名で検索
    location = geolocator.geocode(name)  # ジオコーディング実行
    # 検索結果が得られたときのみ表示
    if location is not None:
        print(f'{rec["W01_001"]}ダム  Nominatim座標={location.latitude, location.longitude} Shapefile座標={shp.points[0][1], shp.points[0][0]}')

# %% 処理11: 逆ジオコーディング（geopy + Nominatimを使用）
from geopy.geocoders import Nominatim
# 堤高150m以上のダムに絞る
over150m_indices = [i for i, rec in enumerate(recs_dam) if rec['W01_007'] > 150]
over150m_shps = [shp for i, shp in enumerate(shps_dam) if i in over150m_indices]
over150m_recs = [rec for i, rec in enumerate(recs_dam) if i in over150m_indices]

# Nominatimを指定
geolocator = Nominatim(user_agent="test")
# ダムを走査
for shp, rec in zip(over150m_shps, over150m_recs):
    # 逆ジオコーディング
    point_str = f'{shp.points[0][1]}, {shp.points[0][0]}'  # "緯度, 経度"を指定
    location = geolocator.reverse(point_str)  # 逆ジオコーディング実行
    print(f'{rec["W01_001"]}ダム  {location.address}')

# %% 保存1: ポイントデータ出力（Shapefile）
# 出力用のデータ（堤高100m以上のダム）作成
# 堤高100m以上のインデックス取得
over100m_indices = [i for i, rec in enumerate(recs_dam) if rec['W01_007'] > 100]
# 堤高100m以上のジオメトリと属性取得
over100m_shps = [shp for i, shp in enumerate(shps_dam) if i in over100m_indices]
over100m_recs = [rec for i, rec in enumerate(recs_dam) if i in over100m_indices]

# Shapefileを出力
outpath = './dams_over100m/dams_over100m.shp'
with shapefile.Writer(outpath, encoding='cp932') as w:
    # 属性のフィールドを定義（名前、型、サイズ、精度を入力）
    w.field('ダム名', 'C', 50, 0)
    w.field('堤高', 'F', 12, 6)
    w.field('総貯水量', 'F', 12, 6)
    # ジオメトリと属性値を行ごとに追加
    for shp, rec in zip(over100m_shps, over100m_recs):
        # ジオメトリ（ポイント）を追加
        w.point(shp.points[0][0], shp.points[0][1])
        # 属性値を追加
        attr = (f'{rec["W01_001"]}ダム', rec['W01_007'], rec['W01_010'])
        w.record(*attr)

# プロジェクションファイル作成 (EPSG:4612)
with open('./dams_over100m/dams_over100m.prj',  'w') as prj:
    epsg = 'GEOGCS["JGD2000",DATUM["Japanese_Geodetic_Datum_2000",SPHEROID["GRS 1980",6378137,298.257222101]],TOWGS84[0,0,0,0,0,0,0],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    prj.write(epsg)

# %% 保存1: ポイントデータ出力（GeoJSON）
import json
import geojson
# 出力用のデータ（堤高100m以上のダム）作成
# 堤高100m以上のインデックス取得
over100m_indices = [i for i, rec in enumerate(recs_dam) if rec['W01_007'] > 100]
# 堤高100m以上のジオメトリと属性取得
over100m_shps = [shp for i, shp in enumerate(shps_dam) if i in over100m_indices]
over100m_recs = [rec for i, rec in enumerate(recs_dam) if i in over100m_indices]

# GeoJSONを出力
outpath = './dams_over100m.geojson'
with open(outpath, 'w', encoding='cp932') as w:
    feature_list = []  # Featureのリスト
    # ジオメトリと属性値を行ごとに追加
    for i, (shp, rec) in enumerate(zip(over100m_shps, over100m_recs)):  # enumerateでIDを付けた方があとで検索しやすい
        # ジオメトリ（ポイント）を作成
        point = geojson.Point((shp.points[0][0], shp.points[0][1]))
        # 属性値を作成
        attr = {'ダム名': f'{rec["W01_001"]}ダム',
                '堤高': float(rec['W01_007']),
                '総貯水量': float(rec['W01_010'])
                }
        # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
        feature = geojson.Feature(geometry=point, id=i, properties=attr)
        feature_list.append(feature)
    # FeatureのリストをFeatureCollectionに変換
    feature_collection = geojson.FeatureCollection(feature_list, 
                                                   crs={'type': 'name',  # 座標系を指定
                                                        'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                        })
    # geojsonファイルに書き出し
    w.write(json.dumps(feature_collection, indent=2))

# %% 保存2: ラインデータ出力（Shapefile）
# 出力用のデータ（上位5位の河川に絞る）作成
# 名称不明と琵琶湖以外のインデックス取得
river_top5_indices = [i for i, rec in enumerate(recs_river) if rec['W05_004'] in ['野洲川', '安曇川', '愛知川', '日野川', '高時川']]
# 名称不明と琵琶湖以外のジオメトリと属性取得
river_top5_shps = [shp for i, shp in enumerate(shps_river) if i in river_top5_indices]
river_top5_recs = [rec for i, rec in enumerate(recs_river) if i in river_top5_indices]

# Shapefileを出力
outpath = './river_top5/river_top5.shp'
with shapefile.Writer(outpath, encoding='cp932') as w:
    # 属性のフィールドを定義（名前、型、サイズ、精度を入力）
    w.field('河川名', 'C', 50, 0)
    w.field('河川コード', 'C', 50, 0)
    # ジオメトリと属性値を行ごとに追加
    for shp, rec in zip(river_top5_shps, river_top5_recs):
        # ジオメトリ（ライン）を追加
        w.line([shp.points])
        # 属性値を追加
        attr = (rec['W05_004'], rec['W05_002'])
        w.record(*attr)

# プロジェクションファイル作成 (EPSG:4612)
with open('./river_top5/river_top5.prj',  'w') as prj:
    epsg = 'GEOGCS["JGD2000",DATUM["Japanese_Geodetic_Datum_2000",SPHEROID["GRS 1980",6378137,298.257222101]],TOWGS84[0,0,0,0,0,0,0],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    prj.write(epsg)

# %% 保存2: ラインデータ出力（GeoJSON）
import json
import geojson
# 出力用のデータ（上位5位の河川に絞る）作成
# 名称不明と琵琶湖以外のインデックス取得
river_top5_indices = [i for i, rec in enumerate(recs_river) if rec['W05_004'] in ['野洲川', '安曇川', '愛知川', '日野川', '高時川']]
# 名称不明と琵琶湖以外のジオメトリと属性取得
river_top5_shps = [shp for i, shp in enumerate(shps_river) if i in river_top5_indices]
river_top5_recs = [rec for i, rec in enumerate(recs_river) if i in river_top5_indices]

# GeoJSONを出力
outpath = './river_top5.geojson'
with open(outpath, 'w', encoding='cp932') as w:
    feature_list = []  # Featureのリスト
    # ジオメトリと属性値を行ごとに追加
    for i, (shp, rec) in enumerate(zip(river_top5_shps, river_top5_recs)):  # enumerateでIDを付けた方があとで検索しやすい
        # ジオメトリ（ポイント）を作成
        line = geojson.LineString(shp.points)
        # 属性値を作成
        attr = {'河川名': rec['W05_004'],
                '河川コード': rec['W05_002']
                }
        # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
        feature = geojson.Feature(geometry=line, id=i, properties=attr)
        feature_list.append(feature)
    # FeatureのリストをFeatureCollectionに変換
    feature_collection = geojson.FeatureCollection(feature_list, 
                                                   crs={'type': 'name',  # 座標系を指定
                                                        'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                        })
    # geojsonファイルに書き出し
    w.write(json.dumps(feature_collection, indent=2))

# %% 保存3: ポリゴンデータ出力（Shapefile）
# 出力用のデータ（面積100km2以上の湖沼）作成
# UTM座標変換
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)
polys_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_lake]
# shp.partsに基づきポリゴン点を一括分割
parts_list = [[i for i in shp.parts] + [len(shp.points) - 1] for shp in shps_lake]
points_hole_list = [[poly[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)] for poly, parts in zip(polys_utm, parts_list)]
# 面積100km2以上のインデックス取得
areas = [Polygon(points_hole[0], points_hole[1:]).area/1000000 for points_hole in points_hole_list]
lake_over100km2_indices = [i for i, area in enumerate(areas) if area > 100]
areas_over100km2 = [area for i, area in enumerate(areas) if i in lake_over100km2_indices]
# 面積100km2以上のジオメトリと属性取得
lake_over100km2_shps = [shp for i, shp in enumerate(shps_lake) if i in lake_over100km2_indices]
lake_over100km2_recs = [rec for i, rec in enumerate(recs_lake) if i in lake_over100km2_indices]

# Shapefileを出力
outpath = './lake_over100km2/lake_over100km2.shp'
with shapefile.Writer(outpath, encoding='cp932') as w:
    # 属性のフィールドを定義（名前、型、サイズ、精度を入力）
    w.field('湖沼名', 'C', 50, 0)
    w.field('最大水深', 'F', 12, 6)
    w.field('面積', 'F', 12, 6)
    # ジオメトリと属性値を行ごとに追加
    for shp, rec, area in zip(lake_over100km2_shps, lake_over100km2_recs, areas_over100km2):
        # shp.partsに基づきポリゴン点を分割（穴を定義）
        parts = [i for i in shp.parts] + [len(shp.points) - 1]
        points_hole = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        # ジオメトリ（ポリゴン）を追加
        w.poly(points_hole)
        # 属性値を追加
        attr = (rec['W09_001'], rec['W09_003'], area)
        w.record(*attr)

# プロジェクションファイル作成 (EPSG:4612)
with open('./lake_over100km2/lake_over100km2.prj', 'w') as prj:
    epsg = 'GEOGCS["JGD2000",DATUM["Japanese_Geodetic_Datum_2000",SPHEROID["GRS 1980",6378137,298.257222101]],TOWGS84[0,0,0,0,0,0,0],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    prj.write(epsg)

# %% 保存3: ポリゴンデータ出力（GeoJSON）
import json
import geojson
# 出力用のデータ（面積100km2以上の湖沼）作成
# UTM座標変換
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)
polys_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_lake]
# shp.partsに基づきポリゴン点を一括分割
parts_list = [[i for i in shp.parts] + [len(shp.points) - 1] for shp in shps_lake]
points_hole_list = [[poly[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)] for poly, parts in zip(polys_utm, parts_list)]
# 面積100km2以上のインデックス取得
areas = [Polygon(points_hole[0], points_hole[1:]).area/1000000 for points_hole in points_hole_list]
lake_over100km2_indices = [i for i, area in enumerate(areas) if area > 100]
areas_over100km2 = [area for i, area in enumerate(areas) if i in lake_over100km2_indices]
# 面積100km2以上のジオメトリと属性取得
lake_over100km2_shps = [shp for i, shp in enumerate(shps_lake) if i in lake_over100km2_indices]
lake_over100km2_recs = [rec for i, rec in enumerate(recs_lake) if i in lake_over100km2_indices]

# GeoJSONを出力
outpath = './lake_over100km2.geojson'
with open(outpath, 'w', encoding='cp932') as w:
    feature_list = []  # Featureのリスト
    # ジオメトリと属性値を行ごとに追加
    for i, (shp, rec, area) in enumerate(zip(lake_over100km2_shps, lake_over100km2_recs, areas_over100km2)):  # enumerateでIDを付けた方があとで検索しやすい
        # shp.partsに基づきポリゴン点を分割（穴を定義）
        parts = [i for i in shp.parts] + [len(shp.points) - 1]
        points_hole = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
        # ジオメトリ（ポイント）を作成
        poly = geojson.Polygon(points_hole)
        # 属性値を作成
        attr = {'湖沼名': rec['W09_001'],
                '最大水深': float(rec['W09_003']),
                '面積': area
                }
        # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
        feature = geojson.Feature(geometry=poly, id=i, properties=attr)
        feature_list.append(feature)
    # FeatureのリストをFeatureCollectionに変換
    feature_collection = geojson.FeatureCollection(feature_list, 
                                                   crs={'type': 'name',  # 座標系を指定
                                                        'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                        })
    # geojsonファイルに書き出し
    w.write(json.dumps(feature_collection, indent=2))

# %% 表示1：ポイントデータ表示（follium）
import folium
import geojson
# 表示用のデータ（堤高100m以上のダム）作成
over100m_indices = [i for i, rec in enumerate(recs_dam) if rec['W01_007'] > 100]
over100m_shps = [shp for i, shp in enumerate(shps_dam) if i in over100m_indices]
over100m_recs = [rec for i, rec in enumerate(recs_dam) if i in over100m_indices]

# 表示用のGeoJSONファイルを作成（出力1と同じ）
feature_list = []  # Featureのリスト
for i, (shp, rec) in enumerate(zip(over100m_shps, over100m_recs)):  # enumerateでIDを付けた方があとで検索しやすい
    # ジオメトリ（ポイント）を作成
    point = geojson.Point((shp.points[0][0], shp.points[0][1]))
    # 属性値を作成
    attr = {'ダム名': f'{rec["W01_001"]}ダム',
            '堤高': float(rec['W01_007']),
            '総貯水量': float(rec['W01_010'])
            }
    # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
    feature = geojson.Feature(geometry=point, id=i, properties=attr)
    feature_list.append(feature)
# FeatureのリストをFeatureCollectionに変換
feature_collection = geojson.FeatureCollection(feature_list, 
                                               crs={'type': 'name',  # 座標系を指定
                                                    'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                    })

# ベースとなる地図を作成
folium_map = folium.Map(location=[35, 135],
                        zoom_start=6)

# GeoJSONファイルを地図に追加
folium.GeoJson(feature_collection, 
               popup=folium.GeoJsonPopup(fields=['ダム名', '堤高']),  # クリック時に表示されるポップアップ
               ).add_to(folium_map)

# 地図を表示
folium_map

# %% 表示2：ラインデータ表示（follium）
# 表示用のデータ（上位5位の河川）作成
river_top5_indices = [i for i, rec in enumerate(recs_river) if rec['W05_004'] in ['野洲川', '安曇川', '愛知川', '日野川', '高時川']]
river_top5_shps = [shp for i, shp in enumerate(shps_river) if i in river_top5_indices]
river_top5_recs = [rec for i, rec in enumerate(recs_river) if i in river_top5_indices]

# 表示用のGeoJSONファイルを作成（出力2と同じ）
feature_list = []  # Featureのリスト
# ジオメトリと属性値を行ごとに追加
for i, (shp, rec) in enumerate(zip(river_top5_shps, river_top5_recs)):  # enumerateでIDを付けた方があとで検索しやすい
    # ジオメトリ（ポイント）を作成
    line = geojson.LineString(shp.points)
    # 属性値を作成
    attr = {'河川名': rec['W05_004'],
            '河川コード': rec['W05_002']
            }
    # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
    feature = geojson.Feature(geometry=line, id=i, properties=attr)
    feature_list.append(feature)
# FeatureのリストをFeatureCollectionに変換
feature_collection = geojson.FeatureCollection(feature_list, 
                                               crs={'type': 'name',  # 座標系を指定
                                                    'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                    })

# ベースとなる地図を作成
folium_map = folium.Map(location=[35.3, 136.1],
                        zoom_start=9)

# GeoJSONファイルを地図に追加
folium.GeoJson(feature_collection, 
               popup=folium.GeoJsonPopup(fields=['河川名', '河川コード']),  # クリック時に表示されるポップアップ
               style_function=lambda feature: {
                    'color': 'blue',  # 線の色
                    'weight': 3,  # 線の太さ
                    'lineOpacity': 0.2,  # 線の不透明度
                    },
               ).add_to(folium_map)

# 地図を表示
folium_map

# %% 出力3：ポリゴンデータ表示（follium）
import folium
import branca.colormap as cm  # カラーマップ用
import geojson
# 出力用のデータ（面積50km2以上の湖沼）作成
# UTM座標変換
src_srs, dst_srs = osr.SpatialReference(), osr.SpatialReference()
src_srs.ImportFromEPSG(4612)
dst_srs.ImportFromEPSG(3099)
trans = osr.CoordinateTransformation(src_srs, dst_srs)
polys_utm = [list(map(lambda point: trans.TransformPoint(point[1], point[0])[:2], shp.points)) for shp in shps_lake]
# shp.partsに基づきポリゴン点を一括分割
parts_list = [[i for i in shp.parts] + [len(shp.points) - 1] for shp in shps_lake]
points_hole_list = [[poly[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)] for poly, parts in zip(polys_utm, parts_list)]
# 面積50km2以上のインデックス取得
areas = [Polygon(points_hole[0], points_hole[1:]).area/1000000 for points_hole in points_hole_list]
lake_over50km2_indices = [i for i, area in enumerate(areas) if area > 50]
areas_over50km2 = [area for i, area in enumerate(areas) if i in lake_over50km2_indices]
# 面積50km2以上のジオメトリと属性取得
lake_over50km2_shps = [shp for i, shp in enumerate(shps_lake) if i in lake_over50km2_indices]
lake_over50km2_recs = [rec for i, rec in enumerate(recs_lake) if i in lake_over50km2_indices]

# 表示用のGeoJSONファイルを作成（出力3と同じ）
feature_list = []  # Featureのリスト
for i, (shp, rec, area) in enumerate(zip(lake_over50km2_shps, lake_over50km2_recs, areas_over50km2)):  # enumerateでIDを付けた方があとで検索しやすい
    # shp.partsに基づきポリゴン点を分割（穴を定義）
    parts = [i for i in shp.parts] + [len(shp.points) - 1]
    points_hole = [shp.points[parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    # ジオメトリ（ポイント）を作成
    poly = geojson.Polygon(points_hole)
    # 属性値を作成
    attr = {'湖沼名': rec['W09_001'],
            '最大水深': float(rec['W09_003']),
            '面積': area
            }
    # Feature（ジオメトリと属性をまとめたもの）を作成してリストに追加
    feature = geojson.Feature(geometry=poly, id=i, properties=attr)
    feature_list.append(feature)
# FeatureのリストをFeatureCollectionに変換
feature_collection = geojson.FeatureCollection(feature_list, 
                                               crs={'type': 'name',  # 座標系を指定
                                                    'properties': {'name': 'urn:ogc:def:crs:EPSG::4612'}
                                                    })

# ベースとなる地図を作成
folium_map = folium.Map(location=[43, 143],  # 初期表示位置
                        zoom_start=7)  # 初期表示倍率

# 塗りつぶし用のカラーマップを作成(https://www.monotalk.xyz/blog/python-folium%E3%81%AE%E3%82%B3%E3%83%AD%E3%83%97%E3%83%AC%E3%82%B9%E5%9B%B3%E3%81%A7%E9%81%B8%E6%8A%9E%E5%8F%AF%E8%83%BD%E3%81%AAfill_color-%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/)
colorscale = cm.linear.YlGnBu_09.scale(0, 300)

# GeoJSONファイルを地図に追加
folium.GeoJson(feature_collection, 
               popup=folium.GeoJsonPopup(fields=['湖沼名', '面積', '最大水深']),  # クリック時に表示されるポップアップ
               style_function=lambda feature: {
                    'fillColor': colorscale(feature['properties']['最大水深']),  # 塗りつぶし色（lambdaで属性値に基づいて色分け可能）
                    'color': 'gray', # 外枠線の色
                    'weight': 1,  # 外枠線の太さ
                    'fillOpacity': 0.8,  # 塗りつぶしの不透明度
                    'lineOpacity': 0.6,  # 外枠線の不透明度
                    },
               ).add_to(folium_map)

# 地図を表示
folium_map
# %%
