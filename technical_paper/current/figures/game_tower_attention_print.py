"""Print-ready vector schematic of the supplied attention-pooling mechanism.

Adapted from the uploaded game_tower_slide.py, with illustrative weight values
removed and separate arrows for values, weights, and raw inputs made explicit.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(16/2.54,7.2/2.54))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
ax.set(xlim=(0,16),ylim=(0,7.2)); ax.axis('off')
NAVY='#18384b'; TEAL='#167f86'; INK='#243b4a'; GRAY='#5a7180'
PALE='#eaf5f4'; LIGHT='#f0f5f7'; WHITE='white'

def box(x,y,w,h,title,sub='',fill=LIGHT,edge=NAVY,sub_size=9,title_size=9.3,dash=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.10',
        linewidth=.9,edgecolor=edge,facecolor=fill,zorder=3,
        linestyle=(0,(3,2)) if dash else '-'))
    if sub:
        ax.text(x+w/2,y+h*.75,title,ha='center',va='center',weight='bold',color=INK,
                fontsize=title_size,zorder=4)
        ax.text(x+w/2,y+h*.32,sub,ha='center',va='center',color=INK,fontsize=sub_size,
                linespacing=1.2,zorder=4)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',color=INK,
                fontsize=title_size,zorder=4)

def line(points,color=GRAY,dashed=False):
    ax.plot(*zip(*points),lw=.95,color=color,zorder=1,
            linestyle=(0,(3,2)) if dashed else '-')

def arrow(a,b,color=GRAY):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
        linewidth=.95,color=color,zorder=2,shrinkA=0,shrinkB=1.8))

ax.text(1.8,6.69,'Game records',ha='center',va='center',fontsize=9.6,
        weight='bold',color=INK)
ax.text(1.8,6.28,r'$\boldsymbol{x}_g\in\mathbb{R}^{34}$',ha='center',va='center',fontsize=9.6,color=INK)
labels=[r'Game $N$',r'Game $N-1$',r'$\ldots$',r'Game $1$']
ys=[5.46,4.76,4.06,3.36]
for label,y in zip(labels,ys):
    box(.3,y,3.0,.55,label,fill=LIGHT)
    line([(3.3,y+.275),(3.7,y+.275)])
box(.3,2.66,3.0,.55,'Padded slot',fill='#f3f4f5',edge='#a3adb4',dash=True)
line([(3.3,2.935),(3.7,2.935)])
line([(3.7,2.935),(3.7,5.735)])
arrow((3.7,5.35),(4.2,5.35))
arrow((3.7,3.25),(4.2,3.25))

box(4.2,4.70,3.6,1.3,'Shared projection',
    r'$34\rightarrow64\rightarrow64$'+'\n'+r'$\boldsymbol{e}_g=f_\phi(\boldsymbol{x}_g)$',fill=PALE,edge=TEAL)
box(4.2,2.60,3.6,1.3,'Scalar score',
    r'$a_g=s_\vartheta([\boldsymbol{e}_g;\boldsymbol{x}_g])$'+'\nShared across games',sub_size=8.5)
box(8.6,2.60,3.8,1.3,'Masked softmax',
    'Exclude padded slots\nNormalize over games',sub_size=8.8)
box(8.6,4.70,3.8,1.3,'Weighted sum',
    r'$\sum_g\alpha_g\boldsymbol{e}_g$',fill=PALE,edge=TEAL,sub_size=11)
box(13.2,4.70,2.5,1.3,'Summary',
    r'$\boldsymbol{h}^{A}\in\mathbb{R}^{64}$',fill=PALE,edge=TEAL,sub_size=10)

# Raw x reaches both the projection and scalar scorer through the left bus.
# Learned e reaches BOTH scalar scoring and weighted aggregation.
arrow((6.0,4.70),(6.0,3.90))
ax.text(6.20,4.29,r'$\boldsymbol{e}_g$',ha='left',va='center',fontsize=9,color=GRAY)
arrow((7.8,5.35),(8.6,5.35))
ax.text(8.2,5.66,r'$\boldsymbol{e}_g$',ha='center',va='center',fontsize=9,color=GRAY)
arrow((7.8,3.25),(8.6,3.25))
ax.text(8.2,3.56,r'$a_g$',ha='center',va='center',fontsize=9,color=GRAY)
arrow((10.5,3.90),(10.5,4.70))
ax.text(10.70,4.30,r'$\alpha_g$',ha='left',va='center',fontsize=9,color=GRAY)
arrow((12.4,5.35),(13.2,5.35))

# The batch validity mask has its own path; padding receives exactly zero mass.
line([(1.8,2.66),(1.8,2.12),(10.5,2.12)])
arrow((10.5,2.12),(10.5,2.60))
ax.text(6.0,1.82,r'Validity mask: padded slots have $\alpha_g=0$',ha='center',va='center',
        fontsize=8.8,color=GRAY)

ax.text(8.0,1.30,'Raw inputs: box scores, exposure, opponent context, and time.',
        ha='center',va='center',fontsize=9,color=INK)
ax.text(8.0,.81,'Time features: days before target and days into season.',
        ha='center',va='center',fontsize=9,color=INK)
ax.text(8.0,.32,'One score per game; no destination query or pairwise game attention.',
        ha='center',va='center',fontsize=9,color=INK)
fig.savefig(OUT/'game_tower_attention_print.pdf',facecolor='white')
fig.savefig(OUT/'game_tower_attention_print.png',facecolor='white',dpi=300)
plt.close(fig)
