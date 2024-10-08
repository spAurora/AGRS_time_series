import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def ndvi_rs(filepath):
    df = pd.read_csv(filepath, header=1)
    siis = df.iloc[:, 3]
    time = []

    def jisuan(pp, time, ttime):
        zz = {}
        sstime = []

        for i in ttime:
            sstime.append(i.strip("[ ").replace("'", '').replace("\n", ""))
        for j in range(len(pp)):
            ii = int(pp[j].strip("[ ]").replace("'", '').replace("\\n", ""))
            if ii == 0:
                ii = np.nan
            if len(sstime) == len(time):
                zz[time[j]] = ii
            else:
                for i in range(len(time)):
                    if time[i] in sstime:
                        zz[time[j]] = ii
                    else:
                        zz[time[j]] = np.nan
        return zz

    doy = siis.iloc[0].split(',')[0::11]
    for i in doy:
        time.append(i.strip("[ ").replace("'", '').replace("\n", ""))
    a = len(time)
    #     print(a)
    lenss = len(siis)
    for i in range(lenss):
        timee = []
        doy = siis.iloc[i].split(',')[0::11]
        for i in doy:
            timee.append(i.strip("[ ").replace("'", '').replace("\n", ""))
        if a <= len(timee):
            time = timee
            a = len(timee)
    print(time)
    blue = pd.DataFrame(columns=time)
    red = pd.DataFrame(columns=time)
    B5 = pd.DataFrame(columns=time)
    B6 = pd.DataFrame(columns=time)
    B7 = pd.DataFrame(columns=time)
    B8a = pd.DataFrame(columns=time)
    nir = pd.DataFrame(columns=time)
    B11 = pd.DataFrame(columns=time)

    for i in range(lenss):
        if i % 500 == 0:
            print(i)
        ttime = siis.iloc[i].split(",")[0::11]

        bb = siis.iloc[i].split(",")[1::11]
        blue = blue.append(jisuan(bb, time, ttime), ignore_index=True)
        rr = siis.iloc[i].split(",")[2::11]
        red = red.append(jisuan(rr, time, ttime), ignore_index=True)

        b5 = siis.iloc[i].split(",")[3::11]
        B5 = B5.append(jisuan(b5, time, ttime), ignore_index=True)

        b6 = siis.iloc[i].split(",")[4::11]
        B6 = B6.append(jisuan(b6, time, ttime), ignore_index=True)

        b7 = siis.iloc[i].split(",")[5::11]
        B7 = B7.append(jisuan(b7, time, ttime), ignore_index=True)

        b8a = siis.iloc[i].split(",")[7::11]
        B8a = B8a.append(jisuan(b8a, time, ttime), ignore_index=True)

        nn = siis.iloc[i].split(",")[6::11]
        nir = nir.append(jisuan(nn, time, ttime), ignore_index=True)

        b11 = siis.iloc[i].split(",")[10::11]
        B11 = B11.append(jisuan(b11, time, ttime), ignore_index=True)

    time = pd.DataFrame(time).T
    time = pd.to_datetime(time.iloc[0])
    time = time.tolist()
    blue = blue.dropna(axis=0, how="all")
    blue = blue.mean(axis=0)
    red = red.dropna(axis=0, how="all")
    red = red.mean(axis=0)
    nir = nir.dropna(axis=0, how="all")
    nir = nir.mean(axis=0)
    B5 = B5.dropna(axis=0, how="all")
    B5 = B5.mean(axis=0)
    B6 = B6.dropna(axis=0, how="all")
    B6 = B6.mean(axis=0)
    B7 = B7.dropna(axis=0, how="all")
    B7 = B7.mean(axis=0)
    B8a = B8a.dropna(axis=0, how="all")
    B8a = B8a.mean(axis=0)
    B11 = B11.dropna(axis=0, how="all")
    B11 = B11.mean(axis=0)
    ndvi = (nir - red) / (nir + red)

    NDRE2 = (blue - B11) / (blue + B11)
    NDVIRE1 = (B8a - B5) / (B8a + B5)
    result = [time, ndvi.iloc[:], NDRE2.iloc[:], NDVIRE1.iloc[:]]
    print(filepath + "已完成")
    return result
def ndvi_s(filepath):
    df = pd.read_csv(filepath, header=1)
    #         green = green[invalid_index]
    #     red = red[invalid_index]
    #     B5 = B5[invalid_index]
    #     B6 = B6[invalid_index]
    #     B7 = B7[invalid_index]
    #     B8a = B8a[invalid_index]
    #     nir = nir[invalid_index]
    #     B11 = B11[invalid_index]
    blue, red, B5, B6, B7, B8a, nir, B11, time = [], [], [], [], [], [], [], [], []
    siis = df.iloc[:, 3]

    def jisuan(pp):
        zz = []
        for j in pp:
            ii = int(j.strip("[ ]").replace("'", '').replace("\\n", ""))
            if ii == 0:
                ii = np.nan
            zz.append(ii)
        return zz

    doy = siis.iloc[0].split(',')[0::11]
    for i in doy:
        time.append(i.strip("[ ").replace("'", '').replace("\n", ""))
    timee = pd.DataFrame(time).T
    time = pd.to_datetime(timee.iloc[0])
    #     time=time.tolist()
    for i in range(len(siis)):
        #            date = np.array(pd.to_datetime(SITS[0::11],format='%Y%m%d'))
        #     green = np.array(SITS[1::11]).astype('float')
        #     red = np.array(SITS[2::11]).astype('float')
        #     B5 = np.array(SITS[3::11]).astype('float')
        #     B6 = np.array(SITS[4::11]).astype('float')
        #     B7 = np.array(SITS[5::11]).astype('float')
        #     nir = np.array(SITS[6::11]).astype('float')
        #     B8a = np.array(SITS[7::11]).astype('float')
        #     B9 = np.array(SITS[8::11]).astype('float')
        #     B10 = np.array(SITS[9::11]).astype('float')
        #     B11 = np.array(SITS[10::11]).astype('float')
        bb = siis.iloc[i].split(",")[1::11]
        blue.append(jisuan(bb))
        rr = siis.iloc[i].split(",")[2::11]
        red.append(jisuan(rr))
        b5 = siis.iloc[i].split(",")[3::11]
        B5.append(jisuan(b5))
        b6 = siis.iloc[i].split(",")[4::11]
        B6.append(jisuan(b6))
        b7 = siis.iloc[i].split(",")[5::11]
        B7.append(jisuan(b7))
        b8a = siis.iloc[i].split(",")[7::11]
        B8a.append(jisuan(b8a))
        nn = siis.iloc[i].split(",")[6::11]
        nir.append(jisuan(nn))
        b11 = siis.iloc[i].split(",")[10::11]
        B11.append(jisuan(b11))
    blue = pd.DataFrame(blue)
    blue = blue.dropna(axis=0, how="all")
    blue = blue.mean(axis=0)
    red = pd.DataFrame(red)
    red = red.dropna(axis=0, how="all")
    red = red.mean(axis=0)
    B5 = pd.DataFrame(B5)
    B5 = B5.dropna(axis=0, how="all")
    B5 = B5.mean(axis=0)
    B6 = pd.DataFrame(B6)
    B6 = B6.dropna(axis=0, how="all")
    B6 = B6.mean(axis=0)
    B7 = pd.DataFrame(B7)
    B7 = B7.dropna(axis=0, how="all")
    B7 = B7.mean(axis=0)
    B8a = pd.DataFrame(B8a)
    B8a = B8a.dropna(axis=0, how="all")
    B8a = B8a.mean(axis=0)
    nir = pd.DataFrame(nir)
    nir = nir.dropna(axis=0, how="all")
    nir = nir.mean(axis=0)
    B11 = pd.DataFrame(B11)
    B11 = B11.dropna(axis=0, how="all")
    B11 = B11.mean(axis=0)
    NDVI = (nir - red) / (nir + red)
    NDRE2 = (blue - B11) / (blue + B11)
    NDVIRE1 = (B8a - B5) / (B8a + B5)
    print(filepath + "已完成")

    return [time, NDVI, NDRE2, NDVIRE1]

import numpy as np
import matplotlib.pyplot as plt
import datetime
filepath1=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M3//"+str(2)+"_samples.txt"
filepath2=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M3//"+str(3)+"_samples.txt"
filepath3=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M3//"+str(8)+"_samples.txt"
filepath4=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M3//"+str(10)+"_samples.txt"
filepath5=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M5//"+str(2)+"_samples.txt"
filepath6=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M5//"+str(3)+"_samples.txt"
filepath7=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M5//"+str(8)+"_samples.txt"
filepath8=r"E:\YJS\0-code\pre_data\rsl-train\experiment_4\M5//"+str(10)+"_samples.txt"


ndvi1=ndvi_s(filepath1)
ndvi2=ndvi_s(filepath2)
ndvi3=ndvi_s(filepath3)
ndvi4=ndvi_s(filepath4)
ndvi5=ndvi_rs(filepath5)
ndvi6=ndvi_rs(filepath6)
ndvi7=ndvi_rs(filepath7)
ndvi8=ndvi_rs(filepath8)

# 时间
s2023=datetime.datetime.strptime("2023-01", '%Y-%m').date()
e2023=datetime.datetime.strptime("2023-12", '%Y-%m').date()

# 绘制
fig=plt.figure(figsize=(40,15))

plt.subplot(3,8,1)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi1[0], ndvi1[1],c="#00BFFF")
plt.title('S1',fontsize = 25)
plt.ylim(-0.1,1)
plt.xlim(s2023,e2023)
plt.yticks(fontsize = 15)
plt.ylabel("NDVI",fontsize = 25)#y轴标签
plt.xticks([])

plt.subplot(3,8,2)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi5[0], ndvi5[1],c="#F0E68C")
plt.xlim(s2023,e2023)
plt.title('S2',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,3)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi2[0], ndvi2[1],c="#DB7093")
plt.xlim(s2023,e2023)
plt.title('S1',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,4)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi6[0], ndvi6[1],c="#40E0D0")
plt.xlim(s2023,e2023)
plt.title('S2',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,5)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi3[0], ndvi3[1],c="#90EE90")
plt.xlim(s2023,e2023)
plt.title('S1',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,6)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi7[0], ndvi7[1],c="#F4A460")
plt.xlim(s2023,e2023)
plt.title('S2',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,7)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi4[0], ndvi4[1],c="#C0C0C0")
plt.xlim(s2023,e2023)
plt.title('S1',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,8)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi8[0], ndvi8[1],c="#F08080")
plt.xlim(s2023,e2023)
plt.title('S2',fontsize = 25)
plt.ylim(-0.1,1)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,9)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi1[0], ndvi1[2],c="#4169E1")

plt.ylim(-0.5,0.5)
plt.xlim(s2023,e2023)
plt.yticks(fontsize = 15)
plt.ylabel("MNDWI",fontsize = 25)#y轴标签
plt.xticks([])

plt.subplot(3,8,10)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi5[0], ndvi5[2],c="#FFD700")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,11)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi2[0], ndvi2[2],c="#FF1493")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,12)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi6[0], ndvi6[2],c="#20B2AA")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,13)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi3[0], ndvi3[2],c="#32CD32")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,14)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi7[0], ndvi7[2],c="#FF8C00")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,15)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi4[0], ndvi4[2],c="#808080")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])


plt.subplot(3,8,16)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi8[0], ndvi8[2],c="#CD5C5C")
plt.xlim(s2023,e2023)

plt.ylim(-0.5,0.5)
plt.xticks([])
plt.yticks([])

plt.subplot(3,8,17)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi1[0], ndvi1[3],c="#000080")
# plt.title('winter wheat',fontsize = 15)
plt.ylim(-0.1,1)
plt.xlim(s2023,e2023)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 12,rotation=50)
plt.ylabel("NDVIRE1",fontsize = 25)#y轴标签

plt.subplot(3,8,18)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi5[0], ndvi5[3],c="#DAA520")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)


plt.subplot(3,8,19)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi2[0], ndvi2[3],c="#C71585")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)


plt.subplot(3,8,20)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi6[0], ndvi6[3],c="#008B8B")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)

plt.subplot(3,8,21)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi3[0], ndvi3[3],c="#228B22")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)

plt.subplot(3,8,22)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi7[0], ndvi7[3],c="#D2691E")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)

plt.subplot(3,8,23)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi4[0], ndvi4[3],c="#000000")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)

plt.subplot(3,8,24)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.scatter(ndvi8[0], ndvi8[3],c="#8B0000")
# plt.title('winter wheat',fontsize = 15)
plt.xlim(s2023,e2023)
plt.ylim(-0.1,1)
plt.yticks([])
plt.xticks(fontsize = 12,rotation=50)
plt.savefig(r'C:\Users\13607\Desktop\BISHEZHITU\shixu2.png', dpi=600)  # 设置dpi和保存文件名
plt.show()
