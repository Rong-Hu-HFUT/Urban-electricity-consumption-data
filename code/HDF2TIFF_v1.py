
import os,re
from osgeo import gdal

def trigger(trigger_flag=0,target_tile=[]):

    dirs = os.listdir(inputFolder)
    list1=[x for x in dirs if os.path.splitext(x)[1]=='.h5']
    list1=[x for x in list1 if x[-3:]=='.h5']
       
    list2=[]
    for tile in target_tile:
        list2.extend([x for x in dirs if tile in os.path.splitext(x)[0]])    
      
    rasterFiles=[x for x in list1 if x in list2]

    

    if trigger_flag!=0:
        print('\n***',str(len(rasterFiles)),'hdf5 files are found ***\n')
        [read_raster(x,layer=select_layer,outputall=outputall_flag,simple_name=simple_name_flag,clip=clip_flag) for x in rasterFiles]
        print('\n*** Process complete ***')
        
        
    elif trigger_flag==0:
        hdflayer = gdal.Open(inputFolder+'\\'+rasterFiles[0], gdal.GA_ReadOnly)
        SubDatasets_list=hdflayer.GetSubDatasets()    
        print('\n***List of band***')
        [print(str(a)+':'+SubDatasets_list[a][0][str.rfind(SubDatasets_list[a][0],'/')+1:]) for a in range(len(SubDatasets_list))]
                     
def read_raster(rasterFiles,layer=[],outputall=0,simple_name=1,clip=0):  

    
    def out_raster(subhdflayer,layer_id=0):
        rlayer = gdal.Open(subhdflayer, gdal.GA_ReadOnly)

        outputName = subhdflayer[str.rfind(subhdflayer,'/')+1:]+'_'
        outputName = outputName.strip().replace(" ","_").replace("/","_").replace("-", "_")
        outputName_Full = outputName + rasterFilePre.replace('.', '_')+ fileExtension

        ind=[i.start() for i in re.finditer('\.', rasterFilePre)]
        outputName_Simple = outputName + rasterFilePre[ind[0]+1:ind[2]].replace('.', '_')+ fileExtension

        if simple_name==1:
            outputName_Full=outputName_Simple

        outputRaster_Full = tempFolder + '\\'+outputName_Full

        HorizontalTileNumber = int(rlayer.GetMetadata_Dict()["HorizontalTileNumber"])
        VerticalTileNumber = int(rlayer.GetMetadata_Dict()["VerticalTileNumber"])

        WestBoundCoord = (10*HorizontalTileNumber) - 180
        NorthBoundCoord = 90-(10*VerticalTileNumber)
        EastBoundCoord = WestBoundCoord + 10
        SouthBoundCoord = NorthBoundCoord - 10
        #projection
        EPSG = "-a_srs EPSG:4326" #WGS84
        translateOptionText = EPSG+" -a_ullr " + str(WestBoundCoord) + " " + str(NorthBoundCoord) + " " + str(EastBoundCoord) + " " + str(SouthBoundCoord)
        translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine(translateOptionText))
        gdal.Translate(outputRaster_Full,rlayer, options=translateoptions)
        
        if clip==1:
            outputRaster_clip=outputRaster_Full
            outputRaster_clip=outputRaster_clip[:-4]+'_clip.tif'      
            dstNodata_val=65535     
            
            if layer_id in [3,4,6]:
                dstNodata_val=255
            OutTile = gdal.Warp(outputRaster_clip,
                                outputRaster_Full, 
                                cutlineDSName=path_shapefile,
                                cropToCutline=True,
                                dstNodata = dstNodata_val)
            OutTile=None
            os.remove(outputRaster_Full)
            outputRaster_Full=outputRaster_clip
           
        target_path=outputFolder_city + '\\'+str(layer_id)+outputName_Full
        dataset = gdal.Open(outputRaster_Full)
        driver = gdal.GetDriverByName('GTiff')
        driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS=LZW"])
        dataset=None
        driver=None
        os.remove(outputRaster_Full)  #lzw压缩数据              
            
    rasterFilePre = rasterFiles[:-3] 
    print(rasterFiles[:-21])
    fileExtension = ".tif"
    hdflayer = gdal.Open(inputFolder+'\\'+rasterFiles, gdal.GA_ReadOnly)
    SubDatasets_list=hdflayer.GetSubDatasets()

    if outputall==0:       
        [out_raster(hdflayer.GetSubDatasets()[i][0],layer_id=i) for i in layer]
        
    else:
        [out_raster(hdflayer.GetSubDatasets()[i][0],layer_id=i) for i in range(0,len(SubDatasets_list))]      

if __name__=='__main__':
    



    inputFolder=r"D:\NBT\20231112"

    tempFolder = r"D:\balckmarble\Temp"

    outputFolder = r"D:\balckmarble\Chinese_Output20231112"

    shape_tiles=r"D:\balckmarble\shape_tile_chinese"

    shape_cities = r"D:\balckmarble\shape_file_chinese"

    trigger_flag=1
    select_layer=[0,1]
    outputall_flag=0
    simple_name_flag=1
    clip_flag=1
    
# *****************************************
    
    list_shp=[x for x in os.listdir(shape_tiles) if os.path.splitext(x)[1]=='.shp']
    
    for tile_city in list_shp:
        target_tile=tile_city[:6]
        city=tile_city[7:-4]
        print('\n'+target_tile,city)
        
        outputFolder_city=outputFolder+ '\\' + city
        if not os.path.exists(outputFolder_city):
            print('create city folder!')
            os.mkdir(outputFolder_city)
        
        path_shapefile=shape_cities+'\\'+city+'.shp'
        trigger(trigger_flag,target_tile=[target_tile])
        if trigger_flag==0:
            break
        

