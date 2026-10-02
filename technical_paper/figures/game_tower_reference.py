"""Publication figure for the proposed pooled international-to-NCAA tower.
Produces a vector PDF and 300 dpi PNG. No empirical attention weights are implied.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                     'ps.fonttype': 42})
fig, ax = plt.subplots(figsize=(16/2.54, 8.4/2.54))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0,16), ylim=(0,8.4)); ax.axis('off')
navy='#18384b'; teal='#167f86'; ink='#243b4a'; gray='#5a7180'
light='#f0f5f7'; pale='#eaf5f4'; white='#ffffff'

def box(x,y,w,h,title,sub='',fill=light,edge=navy,title_size=9.3,sub_size=8.6):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.10',
                    linewidth=0.9,edgecolor=edge,facecolor=fill,zorder=3)
    ax.add_patch(p)
    if sub:
        ax.text(x+w/2,y+h*.72,title,ha='center',va='center',fontsize=title_size,
                weight='bold',color=ink,zorder=4)
        ax.text(x+w/2,y+h*.31,sub,ha='center',va='center',fontsize=sub_size,
                color=ink,linespacing=1.2,zorder=4)
    else:
        ax.text(x+w/2,y+h/2,title,ha='center',va='center',fontsize=title_size,
                color=ink,zorder=4)

def line(points,color=gray,lw=.95,dashed=False):
    ax.plot(*zip(*points), color=color,lw=lw,zorder=1,solid_capstyle='butt',
            linestyle='--' if dashed else '-')

def arrow(a,b,color=gray):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,
                                linewidth=.95,color=color,zorder=2,
                                shrinkA=0,shrinkB=1.5))

# One record set feeds learned per-game vectors and factual evidence summaries.
box(.3,7.38,15.4,.77,'Admissible game records',
    'Box scores  |  Minutes / possessions  |  Competition context  |  Date / age',
    fill=white,title_size=9.4,sub_size=8.8)
box(4.1,5.98,7.2,.86,'Shared game MLP',
    r'Input $\rightarrow 64 \rightarrow 32$; one vector $\boldsymbol{e}_g$ per game',fill=pale,edge=teal)
arrow((7.7,7.38),(7.7,6.84))
line([(14.05,7.38),(14.05,5.28)])
arrow((14.05,5.28),(14.05,4.89))

# Three interpretable aggregation paths plus an evidence path.
centres=[2.0,6.02,10.04,14.05]
for cx in centres[:3]:
    line([(7.7,5.98),(7.7,5.50),(cx,5.50)])
    arrow((cx,5.50),(cx,4.89))
box(.3,3.69,3.4,1.2,'Long-run mean','Exposure-weighted\ngame vectors',sub_size=8.8)
box(4.32,3.69,3.4,1.2,'Recent mean','Exposure + recency\nweighted vectors',sub_size=8.8)
box(8.34,3.69,3.4,1.2,'Time contrasts','Recent - long-run;\nseason contrasts',sub_size=8.8)
box(12.35,3.69,3.4,1.2,'Evidence','Games, exposure,\ncoverage span',fill=white,sub_size=8.8)

# All summary paths enter a single small fusion MLP.
for cx in centres:
    line([(cx,3.69),(cx,3.24),(8.0,3.24)])
arrow((8.0,3.24),(8.0,2.99))
box(4.1,2.12,7.8,.87,'Fuse summaries + evidence',
    r'MLP $\rightarrow 64 \rightarrow 32$: encoded history $\boldsymbol{h}_{iw}$',fill=pale,edge=teal)

# Summary-only seasons have a separate encoder; they are not pseudo-games.
box(.3,2.00,3.4,1.12,'Season summaries','Separate encoder\nPresence flag',
    fill=white,title_size=8.5,sub_size=8.5)
arrow((3.7,2.55),(4.1,2.55))

# Destination variables join downstream: they do not alter the reference pooling.
box(.3,.22,5.0,1.15,'Target context',
    r'Team context, age, signed $\Delta$'+'\nOptional prior-NCAA summary',fill=white,sub_size=8.5)
box(6.1,.22,3.8,1.15,'Shared translator','Condition on history\nand target context',fill=pale,edge=teal,sub_size=8.7)
box(10.7,.22,5.05,1.15,'Season distribution',
    'Games, latent minutes, counts'+'\n'+r'Recorded minutes $\rightarrow$ PER',fill=white,sub_size=8.5)
arrow((8.0,2.12),(8.0,1.37))
arrow((5.3,.79),(6.1,.79))
arrow((9.9,.79),(10.7,.79))
fig.savefig(OUT/'game_tower_reference.pdf',facecolor='white')
fig.savefig(OUT/'game_tower_reference.png',facecolor='white',dpi=300)
plt.close(fig)
