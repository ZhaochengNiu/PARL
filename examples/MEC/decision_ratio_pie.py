import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context='notebook', style='white', font='sans-serif', font_scale=1.0, color_codes=True, rc=None)

Local = np.loadtxt('LocalRatioInEachSlotFilePath25.cache')
Edge = np.loadtxt('EdgeRatioInEachSlotFilePath25.cache')
D2D = np.loadtxt('D2DRatioInEachSlotFilePath25.cache')

Local_temp = sum(Local)
Edge_temp = sum(Edge)
D2D_temp = sum(D2D)

#每个标签占比，会自动按照x/sum(x)所占百分比绘制
data = np.array([Local_temp, Edge_temp, D2D_temp])

labels = ['Local', 'Edge', 'D2D']

#对比各个部分的凸显程度
explode = (0.1, 0.2, 0.3)

plt.pie(data,
        labels=labels, # 设置饼图标签
        colors=["#d5695d", "#5d8ca8", "#65a479"], # 设置饼图颜色
        explode=explode,
        shadow=True,
        autopct="%0.0f%%",
        startangle=70,
        pctdistance=0.5,
       )


# autopct='%.0f%%' 显示百分比
# textprops = {'fontsize':30, 'color':'k'} 大小为30，颜色为黑色
# shadow=True 显示阴影
# startangle，起始角度，0，表示第一块从0开始逆时针转.
# pctdistance，百分比的文本离圆心的距离为

# 显示图例
plt.legend()

plt.savefig('./DecisionRatioPie.pdf', format='pdf')
# 展示
plt.show()
