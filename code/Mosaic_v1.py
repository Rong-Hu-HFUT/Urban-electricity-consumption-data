

import os,re
from osgeo import gdal
import numpy as np

class GRID:
 
  #读图像文件
  def read_img(self,filename):
    dataset=gdal.Open(filename)    #打开文件
 
    im_width = dataset.RasterXSize  #栅格矩阵的列数
    im_height = dataset.RasterYSize  #栅格矩阵的行数
 
    im_geotrans = dataset.GetGeoTransform() #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
 
    del dataset 
    return im_proj,im_geotrans,im_data
 
  #
  def write_img(self,filename,im_proj,im_geotrans,im_data,data_type=16):

    
    datatype = gdal.GDT_UInt16
    if data_type==8:
        datatype=gdal.GDT_Byte
    # dimension
    if len(im_data.shape) == 3:
      im_bands, im_height, im_width = im_data.shape
    else:
      im_bands, (im_height, im_width) = 1,im_data.shape 
 
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
 
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
 
    if im_bands == 1:
      dataset.GetRasterBand(1).WriteArray(im_data)
    else:
      for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
 
    del dataset
    

def get_tile_info(file_name):
    pattern_tile= re.compile(r'h+\d\dv+\d\d')
    tile_name = pattern_tile.findall(file_name)[0]
    pattern_time=re.compile(r'A+\d\d+')
    time_name = pattern_time.findall(file_name)[0]
    time_year=time_name[1:5]
    time_DOY=time_name[5:8]
    layer_id=file_name[0]
    return [layer_id,time_year+time_DOY,tile_name]
    
def mosaic_layer(mosaic_tiles,layer,doy_year):

    path_tiles_full=[path_city_tile+'\\'+ x for x in mosaic_tiles]

    if layer in [3,4,6]:
        data_type=8
        no_data=256
    else:
        data_type=16
        no_data=65536
        
    img1=gdal.Open(path_tiles_full[0],gdal.GA_ReadOnly) 
    rows = img1.RasterYSize
    cols = img1.RasterXSize
    image_out=np.zeros((rows,cols))
    
    for path_img in path_tiles_full:
        img=gdal.Open(path_img,gdal.GA_ReadOnly)
        img_tif=img.GetRasterBand(1).ReadAsArray()
        img_tif=img_tif+1
        idx=np.where(img_tif==no_data)
        img_tif[idx]=0
        image_out=image_out+img_tif
         
    run = GRID()
    proj,geotrans,data = run.read_img(path_tiles_full[0])
    
    
    if not os.path.exists(mosaic_img_output_folder):
        os.mkdir(mosaic_img_output_folder)
    mosaic_img_output_file=mosaic_img_output_folder+'\\'+str(layer)+'A'+str(doy_year)+'.tif'
    
    run.write_img(mosaic_img_output_file,proj,geotrans,image_out,data_type)
    
    
    target_path=mosaic_img_output_folder+'\\'+'L'+str(layer)+'_A'+str(doy_year)+'_'+city+'.tif'
    dataset = gdal.Open(mosaic_img_output_file)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS=LZW"])
    dataset=None
    driver=None
    os.remove(mosaic_img_output_file)
    
    return image_out 

if __name__=='__main__':
    

    path=r"D:\NBT"
    path_city_list=path+'\\'+"tongling"
    path_city_tile_root_folder=path+'\\'+"Output2022"
    mosaic_img_output_root_folder=path+'\\'+'Mosaic2022 tongling'

    select_layer=range(0,1)

    city_list=[x for x in os.listdir(path_city_list) if os.path.splitext(x)[1]=='.shp']
    
    for city_name in city_list:
        city=city_name[:-4]
        print(city)
        path_city_tile=path_city_tile_root_folder+"\\"+city
        mosaic_img_output_folder=mosaic_img_output_root_folder+'\\'+city
        
        layer_tile_list=[x for x in os.listdir(path_city_tile) if os.path.splitext(x)[1]=='.tif']
        layer=list(set([int(x[0]) for x in layer_tile_list]))        
        tile_info=[]
        
        for i in range(len(layer_tile_list)):
            tile_info.append(get_tile_info(layer_tile_list[i]))
        
        tile_info=np.array(tile_info)
        doy_year=list(set(tile_info[:,1].tolist()))
        
        combine_list=[]
        for i in layer:
            list1=np.array((np.where(tile_info[:,0]==str(i))))[0].tolist()
            for j in doy_year:
                print('\nLayerID=',i,'YearDOY=',j)
                list2=np.array(np.where(tile_info[:,1]==str(j)))[0].tolist()
                mosaic_ind=[x for x in list1 if x in list2]
                #print(mosaic_ind)
                mosaic_tiles=[layer_tile_list[x] for x in mosaic_ind]
                print([x[-9:] for x in mosaic_tiles])
                image_out=mosaic_layer(mosaic_tiles,layer=i,doy_year=j)
            
  
