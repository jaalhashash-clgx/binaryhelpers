from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from model_building_helpers import summarize_dist
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.patches as mpatches
from model_building_helpers import analyze_multiple_models
import pandas as pd



def autolabel(rects, ax, loc = 1.02, perc = True, ha = 'center'):
    # attach some text labels

    for rect in rects:
        height = rect.get_height()
        if perc:
            bar_label = '%1.1f' % float(height) + "%"
        else:
            bar_label = '%1.0f' % float(height)
        ax.text(rect.get_x() + rect.get_width()/2., loc*height,
                bar_label,
                ha=ha, va='bottom', fontsize = 16)


def plot_dist_bar(df,column, ax = False, save_fig=None):
    if ax:
        has_x = True
    else:
        has_ax = False

    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
    if not has_ax:
        fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.set_title("{} Distribution".format(column))
    value_counts = (df[column].value_counts()/df.shape[0]).sort_index()*100
    plot = ax.bar(x=value_counts.index,height=value_counts,width=1/(len(value_counts)))
    ax.set_ylabel("Percent of Customers")
    ax.yaxis.set_major_formatter(formatter)
    autolabel(plot,ax)
    if save_fig:
        fig.savefig(save_fig)
    if not has_ax:
        return fig,ax
    else:
        return ax
    
def plot_dist_compare(df,field,target_field,ax=False, save_fig = None, thresh=.001):
    if ax:
        has_x = True
    else:
        has_ax = False
    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
    if not has_ax:
        fig,ax=plt.subplots(1,1,figsize=(10,10))
    ax.set_title("{} vs {} Distribution".format(field,target_field))
    distr = summarize_dist(df,field,target_field,thresh=thresh).sort_index()
    plot = ax.bar(x=np.arange(distr.shape[0]),height=distr['{}_dist'.format(field)]*100,width=1/distr.shape[0],color='lightblue')
    ax2 = ax.twinx()
    plot2 = ax2.plot(np.arange(distr.shape[0]),(distr['{}_dist'.format(target_field)]*100),color = 'orange')

    ax.set_ylabel("Overall Distribution")
    ax.set_xticks(np.arange(distr.shape[0]))
    ax.set_xticklabels(distr.index)
    for i in np.arange(distr.shape[0]):
        xy_tuple = tuple(plot2[0].get_xydata()[i])
        height = xy_tuple[1]
        x = xy_tuple[0]
        bar_label = '%1.1f' % float(height) + "%"
        ax2.text(x,height*1.05,bar_label,horizontalalignment='center')
    ax.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.set_ylabel('{} Rate'.format(target_field))
    ax2.set_ylim(0,100)
    autolabel(plot,ax)
    plt.show()
    if save_fig:
        fig.savefig(save_fig)
    if not has_ax:
        return fig,ax
    else:
        return ax

def plot_dist_plot_compare(data1,data2,field1,field2, labels = None, title = False):

    fig,ax = plt.subplots(1,1,figsize=(8,8))
    if not title:
        ax.set_title('{} Distribution Compare'.format(field1))
    else:
        ax.set_title(title)
    if not labels:
        labels = ['Set1','Set2']
    sns.distplot(data2[field2], ax = ax, label = labels[1],color='dodgerblue')
    sns.distplot(data1[field1],ax=ax, label = labels[0],color='orange')
    model_patch = mpatches.Patch(color='orange',label=labels[0]+': Average = {}'.format(round(data1[field1].mean(),0)))
    Amnat_patch = mpatches.Patch(color='dodgerblue',label=labels[1]+': Average = {}'.format(round(data2[field2].mean(),0)))
    ax.legend(handles = [model_patch,Amnat_patch])
    return fig,ax
    
def compare_dists(df1, df2, tf1, tf2, labels, title = None, savefig=None,thresh=.001):
    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y))
    fig,ax=plt.subplots(1,1,figsize=(10,10))
    if not title:
        ax.set_title("{} vs {} Distribution".format(tf1,tf2))
    else:
        ax.set_title(title)
    unique_1 = df1[tf1].unique()
    unique_2 = df2[tf2].unique()
    missing1 = [item for item in unique_2 if item not in unique_1]
    missing2 = [item for item in unique_1 if item not in unique_2]
    v_counts1 = df1[tf1].value_counts()/df1.shape[0]
    v_counts2 = df2[tf2].value_counts()/df2.shape[0]
    for item in missing1:
        v_counts1[item]=0
    for item in missing2:
        v_counts2[item]=0
    distr1 = (v_counts1.sort_index()*100)
    distr2 = (v_counts2.sort_index()*100)
    diffs = (distr1-distr2).abs().sort_values(ascending=False)[:10]
    distr1 = distr1[diffs.index].sort_index()
    distr2 = distr2[diffs.index].sort_index()
    offset = ((1/len(distr1))/2)*(len(distr1)/3)
   
    plot = ax.bar(np.arange(len(distr1))-offset,height=distr1,width=offset*2,color='lightblue')
    plot2 = ax.bar(np.arange(len(distr1))+offset,height=distr2,width=offset*2,color='orange')
    ax.set_xticks(np.arange(len(distr1)))
    ax.set_xticklabels(distr1.index)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel("Percent of Policies")
    autolabel(plot,ax)
    autolabel(plot2,ax)
    if len(labels)==2:
        blue_patch = mpatches.Patch(color='lightblue', label=labels[0])
        orange_patch = mpatches.Patch(color='orange', label=labels[1])
        ax.legend(handles=[blue_patch,orange_patch])
    else:
        print("Length of Patches does not match length of data no legend")
    if savefig:
        fig.savefig(savefig)
    return ax

def plot_precision_recall(y_actual,y_perc,add_avg = False,add_fill=False,ax=False,**plot_kwargs):
    avg_prec = average_precision_score(y_actual,y_perc)
    prec_rc = pd.concat([pd.Series(val) for val in precision_recall_curve(y_actual,y_perc)],axis=1)
    prec_rc.columns = ['precision','recall','threshold']
    if prec_rc.shape[0]>100000:
        plot_data = prec_rc.iloc[::1000, :]
    elif prec_rc.shape[0]>5000:
        plot_data = prec_rc.iloc[::100, :]
    else:
        plot_data = prec_rc.iloc[::10, :]
    if not ax:
        fig, ax = plt.subplots(figsize=(10,10))
    sns.lineplot(x='recall',y='precision',data=plot_data,ax=ax,**plot_kwargs)
    ax.set_title("Precision vs Recall: AVG PR Score = {0:0.3f}".format(avg_prec),fontsize=20)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if add_fill:
        ax.fill_between(plot_data['recall'],plot_data['precision'],alpha=0.2)
    if add_avg:
        ax.axhline(avg_prec,linestyle='--', color = 'red',alpha=.75)
        ax.text(.3,avg_prec-.02,"Avg Precision Score = {0:0.3f}".format(avg_prec),horizontalalignment='right')
    if not ax:
        return fig,ax
    else:
        return ax
    
def plot_roc_curve(y_actual,y_perc,add_chance=True,add_fill=False,ax=False,use_scores=True,facetwrap=False,**plot_kwargs):
    if use_scores:
        roc_auc = roc_auc_score(y_actual,y_perc)
        prec_rc = pd.concat([pd.Series(val) for val in roc_curve(y_actual,y_perc)],axis=1)
        prec_rc.columns = ['False Positive rate','True Positive Rate','threshold']
        if prec_rc.shape[0]>100000:
            plot_data = prec_rc.iloc[::1000, :]
        elif prec_rc.shape[0]>5000:
            plot_data = prec_rc.iloc[::100, :]
        else:
            plot_data = prec_rc.iloc[::10, :]
    else:
        roc_auc = auc(y_actual,y_perc)
        plot_data = pd.concat([y_actual,y_perc],axis=1)
        plot_data.columns = ['False Positive rate','True Positive Rate']
    if not ax and not facetwrap:
        needs_ax=True
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        needs_ax=False
    if not facetwrap:
        sns.lineplot(x='False Positive rate',y='True Positive Rate',data=plot_data,ax=ax,**plot_kwargs)
    else:
        ax = sns.lineplot(x='False Positive rate',y='True Positive Rate',data=plot_data,**plot_kwargs)
    ax.set_title("ROC Curve:  AUC = {0:0.3f}".format(roc_auc),fontsize=20)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    if add_fill:
        ax.fill_between(plot_data['False Positive rate'],plot_data['True Positive Rate'],y2=plot_data['False Positive rate'],alpha=0.2)
    if add_chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         alpha=.8)
    if facetwrap:
        return ax
    if needs_ax:
        return fig,ax
    else:
        return ax
    
def plot_perc_lift(y_actual,y_perc,use_scores=False,split_num=False,add_baseline = False,ax=False,add_label=True,label_val=.1,facetwrap=False,**plot_kwargs):
    if use_scores:
        base_perc = y_actual.mean()
        df = pd.concat([y_actual,y_perc],axis=1)
        df.columns = ['y_actual','y_perc']
        df = df.sort_values(by='y_perc',ascending=False).reset_index(drop=True)
        if not split_num:
            if df.shape[0]>100000:
                split_num=1000
            elif df.shape[0]>5000:
                split_num=100
            else:
                split_num=10
        means = [df.iloc[:i]['y_actual'].mean() for i in range(split_num,df.shape[0],split_num)]
        lifts = [mean/base_perc for mean in means]
        perc = [i/df.shape[0] for i in range(split_num,df.shape[0],split_num)]
    else:
        perc = y_actual
        lifts = y_perc
    if add_label:
        for percentile,lift in zip(perc,lifts):
            if round(percentile,1)==label_val and round(percentile,2)>=label_val:
                break
        lift_at_10 = lift
        plot_kwargs['label'] = "Lift At {}%: {:,.3f}".format(int(label_val*100),lift_at_10)
    if not ax and not facetwrap:
        needs_ax = True
        fig,ax = plt.subplots(figsize=(10,10))
    else:
        needs_ax = False
    if not facetwrap:
        sns.lineplot(x=perc,y=lifts,ax=ax,**plot_kwargs)
    else:
        ax = sns.lineplot(x=perc,y=lifts,**plot_kwargs)
    if add_baseline:
        ax.axhline(1,linestyle='--', color = 'red',alpha=.75)
    ax.set_xlim(0,1)
    ax.set_ylim(0.8,max(lifts)+.1)
    ax.set_yticks([i for i in np.arange(1,max(lifts)+.1,.2)])
    ax.set_xlabel("Percent of Population")
    ax.set_ylabel("Lift over Average")
    ax.set_title("Lift Chart",fontsize=20)
    vals = ax.get_xticks()
    ax.set_xticklabels(["{:,.0%}".format(x) for x in vals])
    if facetwrap:
        return ax
    if needs_ax:
        return fig,ax
    else:
        return ax
def plot_cum_gains(y_actual,y_perc,cum_perc=False,add_thresh=False,ax=None,return_auc=True,title = "Model Performance: Cumulative Gains",div=1000,add_chance = True,facetwrap=False,**kwargs):
    if not cum_perc:
        data = pd.concat([y_actual,y_perc],axis=1)
        data.columns = ['y_actual','y_perc']
        data['Rank'] = data['y_perc'].rank()
        target_data = data[['y_actual','Rank']].sort_values(by='Rank',ascending=False).reset_index(drop=True)
        target_data['CUM_PERC'] = target_data['y_actual'].cumsum()/target_data['y_actual'].sum()
        target_data['PERC_TOTAL'] = (target_data.index+1)/target_data.shape[0]
    else:
        target_data = pd.concat([y_actual,y_perc],axis=1)
        target_data.columns = ['PERC_TOTAL','CUM_PERC']
        div = 1
    sns.set_style("whitegrid")
    formatter = FuncFormatter(lambda y, pos:"%d%%" % (y*100))
    if return_auc:
        auc_score = auc(target_data['PERC_TOTAL'],target_data['CUM_PERC'])
        kwargs['label'] = "AUC={:,.3f}".format(auc_score)
    if not ax and not facetwrap:
        needs_ax = True
        fig,ax = plt.subplots(figsize=(15,15))
    else:
        needs_ax = False
    if facetwrap:
        ax = sns.lineplot(x='PERC_TOTAL',y='CUM_PERC',data=target_data.iloc[::div, :],**kwargs)
    else:
        sns.lineplot(x='PERC_TOTAL',y='CUM_PERC',data=target_data.iloc[::div, :],ax=ax,**kwargs)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    def get_thresh(thresh):
    	got_thresh=False
    	i=0
    	got_thresh = (target_data['PERC_TOTAL'].round(2)==thresh).any()

    	while not got_thresh and i <5:    			
    		thresh-=.01
    		got_thresh = (target_data['PERC_TOTAL'].round(2)==thresh).any()
    		i+=1
    	new_thresh = target_data[target_data['PERC_TOTAL'].round(2)==thresh]['CUM_PERC'].values.mean().round(2)
    	return new_thresh
    if add_thresh:
        threshes = [get_thresh(thresh) for thresh in add_thresh]


    xticks =list((np.arange(10)+1)/4)
    
    if add_thresh and not facetwrap:
        xticks = sorted(set(xticks+add_thresh))
    ax.set_xticks(xticks)
    if add_thresh and not facetwrap:
        ticks = sorted([.25,.5]+threshes)
        ax.set_yticks(ticks)
    else:
        ax.set_yticks((np.arange(10)+1)/4)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.set_ylim((0,1))
    ax.set_xlim((0,1))
    ax.set_ylabel("Percent Targets Captured",fontsize=18)
    ax.set_xlabel('Percent of Population',fontsize=18)
    ax.set_title(title,fontsize=20)
    if add_chance:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)
    ax.grid(False)

    if add_thresh:
        for x,y in zip(add_thresh,threshes):
            ax.axvline(x,0,y,linestyle='--',linewidth=2,color='orange')
            ax.axhline(y,0,x,linestyle='--',linewidth=2,color='orange')
        if facetwrap:
            ax.text(x,y+.05,"{:.0f}% captured at {:.0f}%".format(y*100,x*100),weight='semibold',wrap=True)
        else:
            i=0
            for tick in ax.yaxis.get_major_ticks():
                if ticks[i] in threshes:
                    tick.label1.set_fontsize(20)
                    tick.label1.set_fontweight('bold')
                i+=1
    if facetwrap:
        return ax
    if needs_ax:
        if return_auc:
            return fig,ax,auc_score
        else:
            return fig,ax
    else:
        if return_auc:
            return auc_score
                
def compare_models(xlsx,model_dict = False,colors = False,return_figs = False,**model_kwargs):

    ## Compares multiple models and returns a dictionary as well as the 3 curves comparing them.
    '''
        Params: 
            xlsx: the excel file you wish to save your data to, supply false if you don't want to save data output
            model_dict: optional supply of a existing model dictionary that contains specified necessary information about the model (minimum is model_dict[model][x][solutions_pred])
            colors: colors you want to use for your plots, if not supplied will use random colors
            plots: what plots you want for your comparison
            return_figs: whether or not to return the figs for your plots
            **model_kwargs: kwarges to pass into analyze multiple models the wrapper around analyze_insp_opt that does all the work to train the model and generate anlaysis data
        returns:
            model_dict: the model analysis dictionary that it created or the one you supplied
            figs: 3 figures that are used to compare models ROC curve, PR curve and Lift chart
    '''


    if not model_dict:
        model_dict = analyze_multiple_models(**model_kwargs)
    solutions_pred_dict = {}
    if xlsx:
        writer = pd.ExcelWriter(xlsx)
        for item in model_dict.keys():
            for target in model_dict[item].keys():
                model_dict[item][target]['summary_df'].to_excel(writer, sheet_name='{}_summary'.format((item.split('_')[-1].replace('Classifier','')+item.split('_')[-2]+target[-2:]+target[:-2])[:25]))
                model_dict[item][target]['ar_curve_df'].to_excel(writer, sheet_name='{}_arcurve'.format((item.split('_')[-1].replace('Classifier','')+item.split('_')[-2]+target[-2:]+target[:-2])[:25]))
                if 'feature_imps' in model_dict[item][target].keys():
                    model_dict[item][target]['feature_imps'].to_excel(writer, sheet_name='{}_feature_imps'.format((item.split('_')[-1].replace('Classifier','')+item.split('_')[-2]+target[-2:]+target[:-2])[:25]))
                if 'perm_imps' in model_dict[item][target].keys():
                    model_dict[item][target]['perm_imps'].to_excel(writer, sheet_name='{}_perm_imps'.format((item.split('_')[-1].replace('Classifier','')+item.split('_')[-2]+target[-2:]+target[:-2])[:25]))
                solutions_pred_dict['{}-{}'.format(item,target)] = model_dict[item][target]['solutions_pred']
                
        writer.save()
    else:
        for item in model_dict.keys():
            for target in model_dict[item].keys():
                solutions_pred_dict['{}-{}'.format(item,target)] = model_dict[item][target]['solutions_pred']
    cnt = len(solutions_pred_dict)
    ##TODO: We want to be able to specify the plots we want in our comparison.  this will be based on the optional argument plots='all'.
    ## For now we will plot all 4 plots (ROC_AUC,PRECISION_RECALL,LIFT, CUM_GAINS)
    fig, ax = plt.subplots(figsize=(10,10))
    fig2,ax2 = plt.subplots(figsize=(10,10))
    fig3,ax3 = plt.subplots(figsize=(10,10))
    fig4,ax4 = plt.subplots(figsize=(10,10))
    patches1 = []
    patches2 = []
    patches3 = []
    patches4 = []
    if not colors:
        colors = ["#"+''.join([('0123456789ABCDEF')[np.random.randint(0,16)] for j in range(6)])
             for i in range(cnt)]
    i=0
    for key in solutions_pred_dict:
        y_actual=solutions_pred_dict[key]['y_actual']
        y_perc = solutions_pred_dict[key]['y_perc']
        
        roc_auc =roc_auc_score(y_actual,y_perc)
        apr = average_precision_score(y_actual,y_perc)
        patches1.append(mpatches.Patch(color=colors[i],label=key.replace('Classifier','')))
        patches2.append(mpatches.Patch(color=colors[i],label='{}: AUC={:,.3f}'.format(key.replace('Classifier',''),roc_auc)))
        patches3.append(mpatches.Patch(color=colors[i],label='{}: MAP={:,.3f}'.format(key.replace('Classifier',''),apr)))
        plot_perc_lift(y_actual,y_perc,use_scores=True,add_baseline=True,linewidth=2,color=colors[i],ax=ax)
        plot_roc_curve(y_actual,y_perc,add_chance=True,linewidth=2,color=colors[i],ax=ax2)
        plot_precision_recall(y_actual,y_perc,add_avg=False,linewidth=2,color=colors[i],ax=ax3)
        auc_score = plot_cum_gains(y_actual,y_perc,color = colors[i],ax=ax4)
        patches4.append(mpatches.Patch(color=colors[i],label='{}: AUC={:,.3f}'.format(key.replace('Classifier',''),auc_score)))
        i+=1
        
    ax4.legend(handles=patches4,title='Model Performance',title_fontsize=20)
    ax4.set_title("Cumulative Gains Model Comparison")
    ax3.legend(handles=patches3,title='Model Performance',title_fontsize=20)
    ax3.set_title("Precision Recall Model Comparison")
    ax2.legend(handles=patches2,title='Model Performance',title_fontsize=20)
    ax2.set_title('ROC Curve Model Comparison')
    ax.legend(handles=patches1, title='Model Performance',title_fontsize=20)
    if return_figs:
        return model_dict, (fig,fig2,fig3,fig4)
    else:
        return model_dict


def look_at_models(data,target,score_list=None,plots_only=True,return_figs=False,**compare_model_kwargs):

    ## Params: Data: data dataframe you are passing to this
    ##          target: the target data you want to measure your comparison with
    ##         score_list: columns from the model that indicator how your model scored

    ##Returns: agg_df: aggregate dataframe that returns the number of target variables in the top 20, bottom 20, top third, middle third and top third of scores
    ##          figs: 3 figures Roc Curve, Lift Chart and Precision recall curve, comparing the performance of the models
    if not score_list:
        score_list = [col for col in data.columns if col.startswith("SCORE")]
    score_dict = {}
    for score in score_list:
        score_dict[score+'_'] = {}
        score_dict[score+'_']['x1'] = {} 
        sols = data[[target,score]].reset_index(drop=True)
        sols.columns = ['y_actual','y_perc']
        score_dict[score+'_']['x1']['solutions_pred'] = sols
    compare_dict, figs = compare_models(False,model_dict=score_dict,return_figs=True,**compare_model_kwargs)
    if not plots_only:
        agg_list = []
        for key in score_dict.keys():
            preds = score_dict[key]['x1']['solutions_pred']
            preds['Rank'] = preds['y_perc'].rank(ascending = False)
            hazard_rate_top100 = preds[preds['Rank']<=preds.shape[0]*(1/5)]['y_actual'].sum()
            hazard_rate_bottom100 = preds[preds['Rank']>=(preds.shape[0]*(4/5))]['y_actual'].sum()
            hazard_rate_top3rd = preds[preds['Rank']<=preds.shape[0]/3]['y_actual'].sum()
            hazard_rate_middle3rd = preds[(preds['Rank']>=preds.shape[0]/3) & (preds['Rank']<preds.shape[0]*(2/3))]['y_actual'].sum()
            hazard_rate_bottom3rd = preds[preds['Rank']>preds.shape[0]*(2/3)]['y_actual'].sum()
            agg_list.append((key,hazard_rate_top100,hazard_rate_bottom100,hazard_rate_top3rd,hazard_rate_middle3rd,hazard_rate_bottom3rd))

        agg_df = pd.DataFrame(agg_list, columns=['Model','Target in Top 20%','Target in Bottom 20%','Target in top 3rd', 'Target in Middle 3rd','Target in bottom 3rd'])
        if return_figs:
            return agg_df, figs
        else:
            return agg_df
    else:
        return figs