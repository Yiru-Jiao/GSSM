'''
This file contains the functions to create the tables and figures
'''

import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import colorcet as cc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src_safety_evaluation.validation_utils.utils_eval_metrics import *


cmap = mpl.colors.LinearSegmentedColormap.from_list('cet_cbl2', cc.CET_CBL2[::-1])
cmap_diverge = mpl.colors.LinearSegmentedColormap.from_list('cet_cbd1', cc.CET_CBD1)
cmap_cycle = mpl.colors.LinearSegmentedColormap.from_list('cet_cbc2', cc.CET_CBC2)
cmap_cw = mpl.colors.LinearSegmentedColormap.from_list('cet_cbtd1', cc.CET_CBTD1)

category = {'SafeBaseline': 'Safe baselines',
            'NearCrash': 'Near-crashes',
            'Crash': 'Crashes',
            'NearCrash-NearCrash': 'Near-crashes',
            'SecondaryNearCrash': 'Near-crashes',
            'NearCrash-CrashRelevant': 'Near-crashes',
            'NearCrash-Crash': 'Crashes',
            'Crash-Crash': 'Crashes',
            'Crash-NearCrash': 'Crashes',
            'CrashRelevant-NearCrash': 'Near-crashes',
            'SecondaryCrash': 'Crashes',
            'NearCrash-OtherConflict': 'Near-crashes',
            'Crash-OtherConflict': 'Crashes',
            'CrashRelevant-Crash': 'Crashes',
            'Crash-CrashRelevant': 'Crashes'}

conflict_type = {'leading': 'Leading',
                 'following': 'Following',
                 'adjacent_lane': 'Adjacent lane',
                 'merging': 'Merging',
                 'turning_into_opposite': 'Crossing/turning',
                 'turning_into_parallel': 'Crossing/turning',
                 'turning_across_opposite': 'Crossing/turning',
                 'turning_across_parallel': 'Crossing/turning',
                 'intersection_crossing': 'Crossing/turning',
                 'pedestrian': 'Pedestrian/cyclist',
                 'cyclist': 'Pedestrian/cyclist',
                 'animal': 'Animal',
                 'oncoming': 'Oncoming',
                 'obstacle': 'Shapeless obstacle',
                 'single': 'Single',
                 'parked': 'Parked',
                 'unknown': 'Unknown',
                 'none': 'None',
                 'nan': 'None'}


def remove_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)
    ax.set_frame_on(False)


def light_color(color, alpha=0.5):
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)
    return tuple([c + (1 - c) * (1 - alpha) for c in color])
    

def highlight_col(col, highlight_type='max', scientific_notation=False, involve_second=False):
    if np.mean(col)>0.01:
        col = np.round(col.values, 3)
    else:
        col = col.values
    if highlight_type == 'max':
        sorted_col = np.sort(col[~np.isnan(col)])[::-1]
    elif highlight_type == 'min':
        sorted_col = np.sort(col[~np.isnan(col)])
    else:
        raise ValueError("highlight_type must be either 'max' or 'min'.")
    
    if len(sorted_col) >= 2:
        is_extreme = col==sorted_col[0]
        is_second_extreme = col==sorted_col[1]
    elif len(sorted_col) == 1:
        is_extreme = col==sorted_col[0]
        is_second_extreme = np.zeros(len(col), dtype=bool)
    else:
        is_extreme = np.zeros(len(col), dtype=bool)
        is_second_extreme = np.zeros(len(col), dtype=bool)
    
    styled_col = []
    if scientific_notation and np.mean(col)<=0.01:
        for i, v in enumerate(col):
            if is_extreme[i]:
                styled_col.append(f'\\textbf{{\\underline{{{v:.3E}}}}}'.replace('E-0','E-'))
            elif is_second_extreme[i] and involve_second:
                styled_col.append(f'\\textbf{{{v:.3E}}}'.replace('E-0','E-'))
            else:
                styled_col.append(f'{v:.3E}'.replace('E-0','E-'))
    else:
        for i, v in enumerate(col):
            if is_extreme[i]:
                styled_col.append(f'\\textbf{{\\underline{{{v:.3f}}}}}')
            elif is_second_extreme[i] and involve_second:
                styled_col.append(f'\\textbf{{{v:.3f}}}')
            else:
                styled_col.append(f'{v:.3f}')
    return styled_col


def highlight(df, max_cols=[], min_cols=[], scientific_notation=False, involve_second=False):
    for col in max_cols:
        df[col] = highlight_col(df[col], 'max', scientific_notation, involve_second)
    for col in min_cols:
        df[col] = highlight_col(df[col], 'min', scientific_notation, involve_second)
    return df


def reconstruction_error(data_ego, data_sur, event_type):
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4), gridspec_kw={'hspace': 0.4, 'wspace': 0.4})
    for row, target, data in zip(range(2), ['Subject', 'Object'], [data_ego, data_sur]):
        if target=='Subject':
            var_list = {'Speed': ['m/s', 'v_ekf','speed_comp'], 'Acceleration': ['m/s$^2$', 'acc_ekf','acc_lon']}
        else:
            var_list = {'Displacement': ['m'], 'Speed': ['m/s', 'v_ekf','speed_comp']}

        for col, key, values in zip(range(2), var_list.keys(), var_list.values()):
            if key == 'Displacement':
                error = ((data['x_ekf']-data['x'])**2 + (data['y_ekf']-data['y'])**2)**0.5
            else:
                error = data[values[1]] - data[values[2]]
            
            mean, std = error.mean(), error.std()
            limits = [mean - 3*std, mean + 3*std] if key != 'Displacement' else [0, mean + 3*std]
            _ = axes[row, col].hist(error, alpha=0.75, bins=np.linspace(limits[0], limits[1], 30))
            axes[row, col].text(0.95, 0.95, f'mean={mean:.4f}\nstd={std:.4f}', ha='right', va='top', transform=axes[row, col].transAxes)
            axes[row, col].set_title(f'{target} {key} error ({values[0]})')
    _ = fig.suptitle(f'{event_type} event reconstruction error distribution')
    return fig


def event_curve(ax, models, colors, conflict_warning, curve_type='prc'):
    line_styles = ['solid', 'dashdot', (0, (3,1,1,1,1,1)), 'dashed', 'dotted']
    for model, color, ls in tqdm(zip(models, colors, line_styles), total=len(models), desc='Plotting curve'):
        if model == 'highD_current':
            zorder = 0
        else:
            zorder = -5
        tp, fp, tn, fn = get_statistics(conflict_warning[conflict_warning['model']==model])
        if curve_type == 'prc':
            precision = tp/(tp + fp) * 100
            recall = tp/(tp + fn) * 100
            # ax.fill_between(recall, precision, color=color, alpha=0.75, lw=0.35, label=model)
            ax.plot(recall, precision, color=color, lw=1., ls=ls, label=model, zorder=zorder)
            ax.set_xlabel('Recall (%)', labelpad=1)
            ax.set_ylabel('Precision (%)', labelpad=0)
        elif curve_type == 'roc':
            fnr = fn/(tp + fn) * 100
            fpr = fp/(fp + tn) * 100
            # ax.fill_between(fpr, fnr, 100, color=color, alpha=0.75, lw=0.35, label=model)
            ax.plot(fpr, 100-fnr, color=color, lw=1., ls=ls, label=model, zorder=zorder)
            ax.set_xlabel('False positive rate (%)', labelpad=1)
            ax.set_ylabel('True positive rate (%)', labelpad=0)
        elif curve_type == 'atc':
            w = conflict_warning[conflict_warning['model']==model].copy()
            w['TTI'] = w['impact_time'] - w['warning_timestamp'] / 1000.0
            median_TTI = w[w['TTI']<10].groupby('threshold')['TTI'].median()
            f1 = tp / (tp + 0.5*(fp + fn))
            # ax.fill_between(median_TTI, f1, color=color, alpha=0.75, lw=0.35, label=model)
            ax.plot(median_TTI, f1.loc[median_TTI.index], 
                    color=color, lw=1., ls=ls, label=model, zorder=zorder)
            ax.set_xlabel('Median time to impact (s)', labelpad=1)
            ax.set_ylabel('F1 score', labelpad=0)


def get_rates(conflict_warning, models, event_meta, events):
    event_meta = event_meta[event_meta['conflict']!='none']
    if events=='all':
        event_ids = event_meta['event_id'].values
    elif events=='rear-end':
        event_ids = event_meta[event_meta['conflict']=='leading']['event_id'].values
    elif events=='lateral':
        event_ids = event_meta[event_meta['conflict'].isin(['adjacent_lane','merging','turning_into_parallel','turning_into_opposite',
                                                            'turning_across_parallel','turning_across_opposite','intersection_crossing'
                                                            'pedestrian','cyclist'])]['event_id'].values
    elif events=='others':
        event_ids = event_meta[~event_meta['conflict'].isin(['leading','adjacent_lane','merging','turning_into_parallel','turning_into_opposite',
                                                             'turning_across_parallel','turning_across_opposite','intersection_crossing'])]['event_id'].values
    else:
        event_ids = event_meta[event_meta['conflict'].isin(events)]['event_id'].values

    table = pd.DataFrame(columns=['model','mixrate','auprc','aroc_80','aroc_90','pprc_80','pprc_90'])
    for model in tqdm(models):
        filtered_warning = conflict_warning[(conflict_warning['model']==model)&(conflict_warning['event_id'].isin(event_ids))]

        if 'mixed' in model:
            mixrate = model.split('_mixed')[1]
        else:
            mixrate = '0.0'

        eval_metrics = get_eval_metrics(filtered_warning, thresholds={'roc': [0.80, 0.90], 'prc':[0.80, 0.90],'tti':None})
        eval_metrics['model'] = model
        eval_metrics['mixrate'] = mixrate
        table.loc[len(table), list(eval_metrics.keys())] = list(eval_metrics.values())
    table[table.columns[1:]] = table[table.columns[1:]].astype(float)
    return table


def draw_data_scalability(conflict_warning, event_meta):
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.75), constrained_layout=True, gridspec_kw={'wspace':0.1, 'width_ratios':[1,1,1.5]})
    colors = cmap_cw([0.01, 0.25, 0.75, 0.99])

    ax = axes[0]
    ax.set_title('(a) Increasing crossings in ArgoverseHV', pad=5)
    ax.set_xlabel('Additional proportion (%)', labelpad=1)
    ax.set_ylabel('Metric value', labelpad=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.05])
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100', ''])
    models = conflict_warning['model'].unique()
    models = [m for m in models if 'mixed' in m and 'ArgoverseHV' in m]
    models = models + ['SafeBaseline_current']
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['turning_into_parallel','turning_into_opposite',
                                                                    'turning_across_parallel','turning_across_opposite',
                                                                    'intersection_crossing','pedestrian','cyclist'])
    table = table.sort_values('mixrate', ascending=True)
    for metric, alpha, marker in zip(['aroc_80','aroc_90','pprc_80','pprc_90','auprc'], [0.25,0.4,0.65,0.8,0.99], ['^','s','<','p','o']):
        ax.plot(table['mixrate'], table[metric], color=light_color(colors[0],alpha), lw=0.5, label=metric,
                marker=marker, markersize=5, markeredgecolor='k', markeredgewidth=0.1)

    ax = axes[1]
    ax.set_title('(b) Increasing lane changes in highD', pad=5)
    ax.set_xlabel('Additional proportion (%)', labelpad=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.05])
    ax.set_xticklabels(['0', '20', '40', '60', '80', '100', ''])
    models = conflict_warning['model'].unique()
    models = [m for m in models if 'mixed' in m and 'highD' in m]
    models = models + ['SafeBaseline_current']
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['adjacent_lane','merging'])
    table = table.sort_values('mixrate', ascending=True)
    for metric, alpha, marker in zip(['aroc_80','aroc_90','pprc_80','pprc_90','auprc'], [0.25,0.4,0.65,0.8,0.99], ['^','s','<','p','o']):
        ax.plot(table['mixrate'], table[metric], color=light_color(colors[-1],alpha), lw=0.5, label=metric,
                marker=marker, markersize=5, markeredgecolor='k', markeredgewidth=0.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ['$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                        '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                        '$\\mathrm{AUPRC}$'],
            loc='upper center', bbox_to_anchor=(0.5, 0.05),
            ncol=5, frameon=False, handlelength=2.5, handletextpad=0.4, columnspacing=1)

    ax = axes[2]
    ax.set_title('(c) Evaluation on lateral interactions', pad=5)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_xlim(-1, 24)
    ax.set_xticks([1.5, 6.5, 11.5, 16.5, 21.5, 24])
    ax.set_xticklabels(['$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                        '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                        '$\\mathrm{AUPRC}$', ''])
    models = [
        'SafeBaseline_current',
        'SafeBaseline_ArgoverseHV_current_mixed0.1',
        'SafeBaseline_highD_current_mixed1.0',
        'SafeBaseline_ArgoverseHV_highD_current'
    ]
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = get_rates(filtered_warning, models, event_meta, events=['turning_into_parallel','turning_into_opposite',
                                                                    'turning_across_parallel','turning_across_opposite',
                                                                    'intersection_crossing','pedestrian','cyclist',
                                                                    'adjacent_lane','merging'])
    xaxis = np.array([0, 5, 10, 15, 20])
    for addition in [0,1,2,3]:
        ax.bar(xaxis+addition, table.loc[addition][['aroc_80','aroc_90','pprc_80','pprc_90','auprc']].values.astype(float),
            color=colors[addition], label='test')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ['S-C', 'SA-C', 'Sh-C', 'SAh-C'],
            loc='lower left', bbox_to_anchor=(0.01, 0.75),
            ncol=2, frameon=False, handlelength=1, handletextpad=0.4, columnspacing=1)

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_ylim(0.5180533071826404, 0.8773854868551544)
        ax.set_yticks([0.6, 0.7, 0.8, 0.8773854868551544])
        ax.set_yticklabels(['0.6', '0.7', '0.8', ''])

    return fig, axes


def draw_feature_scalability(conflict_warning):
    fig, axes = plt.subplots(1, 7, figsize=(7.05, 1.3), sharex=True, constrained_layout=True, gridspec_kw={'wspace':0.1})

    models = [
            'SafeBaseline_current',
            'SafeBaseline_current_environment',
            'SafeBaseline_current_environment_profiles',
            'SafeBaseline_current+acc',
            'SafeBaseline_current+acc_environment',
            'SafeBaseline_current+acc_environment_profiles',
            ]
    filtered_warning = conflict_warning[conflict_warning['model'].isin(models)]
    table = pd.DataFrame(columns=['model','auprc','aroc_80','aroc_90','pprc_80','pprc_90','PTTI_star','mTTI_star'])
    for model in models:
        metrics = get_eval_metrics(conflict_warning[conflict_warning['model']==model],
                                thresholds={'roc': [0.80, 0.90], 'prc':[0.80, 0.90],'tti':1.5})
        metrics['model'] = model
        table.loc[len(table), list(metrics.keys())] = list(metrics.values())
    table[table.columns[1:]] = table[table.columns[1:]].astype(float)
    colors = cmap_cw([0.4, 0.2, 0.05, 0.6, 0.8, 0.95])

    for axid in range(7):
        ax = axes[axid]
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='both', which='both', direction='in')
        ax.set_xlim(-1, 7)
        ax.set_xticks([7])
        ax.set_xticklabels([''])
        if axid!=6:
            ax.set_ylim(0, 0.9369093231162197)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.9369093231162197])
            ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', ''])
        else:
            ax.set_ylim(0, 3.)
            ax.set_yticks([0, 0.5, 1.5, 2.5, 3])
            ax.set_yticklabels(['0', '0.5', '1.5', '2.5', ''])

        ax.bar([0,1,2], table.loc[[0,1,2]][list(metrics.keys())[axid]].values.astype(float),
            color=colors[:3], label='test', hatch='+++', alpha=0.99)
        ax.bar([4,5,6], table.loc[[3,4,5]][list(metrics.keys())[axid]].values.astype(float),
            color=colors[3:], label='test', hatch='xxxx', alpha=0.99)
        
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[0].patches+handles[1].patches,
            ['S-C', 'S-CE', 'S-CET', 'S-Ca', 'S-CaE', 'S-CaET'],
            loc='lower center', bbox_to_anchor=(0.5, -0.15),
            ncol=6, frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=1)

    for axid, title in zip(range(7), ['$\\mathrm{AUPRC}$', '$A_{80\\%}^\\mathrm{ROC}$', '$A_{90\\%}^\\mathrm{ROC}$',
                                        '$\\mathrm{Precision}_{80\\%}^\\mathrm{PRC}$', '$\\mathrm{Precision}_{90\\%}^\\mathrm{PRC}$',
                                        '$P^*_{\\mathrm{TTI}\\geq1.5}$', '$m\\mathrm{TTI}^*$']):
        axes[axid].set_title(title, pad=3)

    return fig, axes


def stacked_bar(ax, values, labels, colors=None, direction='vertical', alpha=1, hatches=None):
    # Adjust values to make sure the sum is 100
    percentages = np.round(values/np.sum(values), 3)
    if percentages.sum() < 1:
        gap = 1 - percentages.sum()
        percentages[np.argmax(values)] += gap
        percentages = np.round(percentages, 3)
    if percentages.sum() > 1:
        gap = percentages.sum() - 1
        percentages[np.argmax(values)] -= gap
        percentages = np.round(percentages, 3)
    
    # Plot stacked bars
    base = 0
    txt = []
    if colors is None:
        colors = cmap(np.linspace(0.15,0.85,len(values)))
    if hatches is None:
        hatches = ['' for _ in range(len(values))]
    for percentage, label, color, hatch in zip(percentages, labels, colors, hatches):
        if direction=='vertical':
            ax.bar(0, percentage, bottom=base, color=color, label=label, width=1, alpha=alpha, hatch=hatch)
            txt.append(ax.text(0, base+percentage/2, f'{percentage*100:.1f}%', ha='center', va='center', color='w'))
        if direction=='horizontal':
            ax.barh(0, percentage, left=base, color=color, label=label, height=1, alpha=alpha, hatch=hatch)
            txt.append(ax.text(base+percentage/2, 0, f'{percentage*100:.1f}%', ha='center', va='center', color='w'))
        base += percentage
    if direction=='vertical':
        ax.set_ylim(0,1)
        ax.set_yticks([])
    if direction=='horizontal':
        ax.set_xlim(0,1)
        ax.set_xticks([])

    # Adjust text position to avoid overlap
    for i in range(1, len(txt)):
        if direction=='vertical':
            if txt[i].get_position()[1] < txt[i-1].get_position()[1]+0.05:
                txt[i].set_position((txt[i].get_position()[0], txt[i-1].get_position()[1]+0.05))
        if direction=='horizontal':
            if txt[i].get_position()[0] < txt[i-1].get_position()[0]+0.05:
                txt[i].set_position((txt[i-1].get_position()[0]+0.05, txt[i].get_position()[1]))


def draw_generalisability(conflict_warning, models, event_meta, voted_events):
    fig = plt.figure(figsize=(7.05, 3.0))

    ax = fig.add_axes([0, 0, 0.25, 1])
    ax.set_title('(a) Event type distribution', pad=5)
    ax_bar = ax.inset_axes([0, 0.75, 1, 0.15], xlim=(0, 1), ylim=(-0.5, 0.5))
    ax_pie = ax.inset_axes([0., -0.2, 1, 0.9], xlim=(0, 1), ylim=(-0.5, 0.5))
    remove_box(ax), remove_box(ax_bar), remove_box(ax_pie)
    bar_data = voted_events.value_counts('conflict')
    pie_data = bar_data.drop(['leading','adjacent_lane'])
    bar_data.loc['Other lateral'] = pie_data.sum()
    bar_data = bar_data.loc[['leading', 'adjacent_lane', 'Other lateral']]
    stacked_bar(ax_bar, bar_data.values, ['Rear-end','Adjacent lane','Other lateral'],
                direction='horizontal', colors=cmap(np.linspace(0.15, 0.5, 4)))
    ax_bar.legend(loc='lower center', bbox_to_anchor=(0.5, 0.85), ncol=3,
                frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)
    turning = ['turning_into_parallel', 'turning_across_opposite', 'turning_across_parallel', 'turning_into_opposite', 'intersection_crossing']
    pie_data.loc['Turning&Crossing'] = pie_data.loc[turning].sum()
    pie_data = pie_data.drop(turning)
    pie_data = pie_data.rename({'merging':'Merging','pedestrian':'Pedestrian','cyclist':'Cyclist','oncoming':'Oncoming',
                                'parked':'Parked','animal':'Animal','unknown':'Unknown'})
    pie_data = pie_data.sort_values(ascending=False)
    patches, _, autopct = ax_pie.pie(pie_data, autopct=lambda p: f'{p*pie_data.sum()/100:.0f}',
                                        pctdistance=0.85, textprops={'color': 'w'}, startangle=270,
                                        colors=cmap(np.linspace(0.4, 0.85, len(pie_data))))
    ax.legend(patches, pie_data.index, loc='lower center', bbox_to_anchor=(0.5, 0.5),
                ncol=2, frameon=False, handlelength=0.7, handletextpad=0.4, columnspacing=0.75)

    event_meta = event_meta[event_meta['conflict']!='none']
    colors = cmap([0.15, 0.3, 0.45])

    ax_roc = fig.add_axes([0.32, 0.7, 0.7, 0.25])
    ax_roc.set_title('(b) ROC curves', pad=15)
    remove_box(ax_roc)
    ax_prc = fig.add_axes([0.32, 0.34, 0.7, 0.25])
    ax_prc.set_title('(c) PRC curves', pad=5)
    remove_box(ax_prc)
    ax_atc = fig.add_axes([0.32, -0.02, 0.7, 0.25])
    ax_atc.set_title('(d) ATC curves', pad=5)
    remove_box(ax_atc)

    def get_inset_axes(col):
        inset_ax_roc = ax_roc.inset_axes([col*(0.18+0.025), 0, 0.18, 1], xlim=(-5, 105), ylim=(-5, 105))
        inset_ax_roc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_roc.set_xticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_yticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_aspect('equal')
        inset_ax_prc = ax_prc.inset_axes([col*(0.18+0.025), 0, 0.18, 1], xlim=(-5, 105), ylim=(-5, 105))
        inset_ax_prc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_prc.set_xticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_yticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_aspect('equal')
        inset_ax_atc = ax_atc.inset_axes([col*(0.18+0.025), 0, 0.18, 1], xlim=(0.5, 4.5), ylim=(0., 1.))
        inset_ax_atc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_atc.set_xticks([1, 2, 3, 4])
        inset_ax_atc.set_yticks([0.3, 0.6, 0.9])
        inset_ax_atc.set_aspect(4./1)
        return inset_ax_roc, inset_ax_prc, inset_ax_atc
    ax_roc_re, ax_prc_re, ax_atc_re = get_inset_axes(0)
    event_curve_list = []
    event_curve_list.append([['leading'], ax_roc_re, ax_prc_re, ax_atc_re, 'Rear-end'])
    ax_roc_adj, ax_prc_adj, ax_atc_adj = get_inset_axes(1)
    event_curve_list.append([['adjacent_lane'], ax_roc_adj, ax_prc_adj, ax_atc_adj, 'Adjacent lane'])
    ax_roc_tap, ax_prc_tap, ax_atc_tap = get_inset_axes(2)
    event_curve_list.append([['turning_into_parallel','turning_across_opposite','turning_into_parallel','turning_into_opposite','intersection_crossing'], 
                            ax_roc_tap, ax_prc_tap, ax_atc_tap, 'Crossing/turning'])
    ax_roc_mer, ax_prc_mer, ax_atc_mer = get_inset_axes(3)
    event_curve_list.append([['merging'], ax_roc_mer, ax_prc_mer, ax_atc_mer, 'Merging'])
    ax_roc_pca, ax_prc_pca, ax_atc_pca = get_inset_axes(4)
    event_curve_list.append([['pedestrian','cyclist','animal'], ax_roc_pca, ax_prc_pca, ax_atc_pca, 'With pedestrian/\ncyclist/animal'])

    def event_curve(ax, models, colors, conflict_warning, curve_type='prc'):
        line_styles = ['solid', 'dashdot', (0, (3,1,1,1,1,1)), 'dashed', 'dotted']
        for model, color, ls in tqdm(zip(models, colors, line_styles), total=len(models), desc='Plotting curve'):
            tp, fp, tn, fn = get_statistics(conflict_warning[conflict_warning['model']==model])
            if curve_type == 'prc':
                precision = tp/(tp + fp) * 100
                recall = tp/(tp + fn) * 100
                # ax.fill_between(recall, precision, color=color, alpha=0.75, lw=0.35, label=model)
                ax.plot(recall, precision, color=color, lw=1., ls=ls, label=model)
            elif curve_type == 'roc':
                fnr = fn/(tp + fn) * 100
                fpr = fp/(fp + tn) * 100
                # ax.fill_between(fpr, fnr, 100, color=color, alpha=0.75, lw=0.35, label=model)
                ax.plot(fpr, 100-fnr, color=color, lw=1., ls=ls, label=model)
            elif curve_type == 'atc':
                w = conflict_warning[conflict_warning['model']==model].copy()
                w['TTI'] = w['impact_time'] - w['warning_timestamp'] / 1000.0
                median_TTI = w[w['TTI']<10].groupby('threshold')['TTI'].median()
                f1 = tp / (tp + 0.5*(fp + fn))
                # ax.fill_between(median_TTI, f1, color=color, alpha=0.75, lw=0.35, label=model)
                ax.plot(median_TTI, f1.loc[median_TTI.index], 
                        color=color, lw=1., ls=ls, label=model)

    for events, inset_ax_roc, inset_ax_prc, inset_ax_atc, title in tqdm(event_curve_list):
        event_ids = event_meta[event_meta['conflict'].isin(events)]['event_id'].values
        filtered_warning = conflict_warning[(conflict_warning['event_id'].isin(event_ids))]

        inset_ax_roc.fill_between([0,80], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10, label='Safety-critical')
        event_curve(inset_ax_roc, models, colors, filtered_warning, curve_type='roc')
        inset_ax_roc.set_xlim(0, 80)
        inset_ax_roc.set_ylim(20, 100)
        inset_ax_roc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_roc.set_xticks([0, 20, 40, 60, 80])
        inset_ax_roc.set_yticks([20, 40, 60, 80, 100])

        inset_ax_prc.fill_betweenx([20,100], 80, 100, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10)
        event_curve(inset_ax_prc, models, colors, filtered_warning, curve_type='prc')
        inset_ax_prc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_prc.set_xlim(20, 100)
        inset_ax_prc.set_ylim(20, 100)
        inset_ax_prc.set_xticks([20, 40, 60, 80, 100])
        inset_ax_prc.set_yticks([20, 40, 60, 80, 100])

        inset_ax_atc.fill_between([0.5,4.5], 0.8, 0.95, fc=light_color('tab:red',0.05), ec=light_color('tab:red',0.35), lw=0.35, zorder=-10)
        event_curve(inset_ax_atc, models, colors, filtered_warning, curve_type='atc')
        inset_ax_atc.tick_params(axis='both', which='both', pad=2, direction='in')
        inset_ax_atc.set_xlim(0.5, 4.5)
        inset_ax_atc.set_ylim(0.45, 0.95)
        inset_ax_atc.set_xticks([1, 2, 3, 4])
        inset_ax_atc.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        inset_ax_atc.set_aspect(4./0.5)

        inset_ax_roc.set_title(title, pad=2, fontsize=7)
        if title!='Rear-end':
            inset_ax_roc.set_yticklabels([])
            inset_ax_prc.set_yticklabels([])
            inset_ax_atc.set_yticklabels([])
        else:
            inset_ax_roc.set_ylabel('True positive rate (%)', labelpad=1)
            inset_ax_prc.set_ylabel('Precision (%)', labelpad=1)
            inset_ax_atc.set_ylabel('F1 score', labelpad=1)
        if title!='Merging' and 'lane' not in title:
            inset_ax_roc.set_xticklabels([])
            inset_ax_prc.set_xticklabels([])
            inset_ax_atc.set_xticklabels([])
        if 'turning' in title:
            inset_ax_roc.set_xlabel('False Positive Rate (%)', labelpad=1)
            inset_ax_prc.set_xlabel('Recall (%)', labelpad=1)
            inset_ax_atc.set_xlabel('Median time to impact (s)', labelpad=1)

    handles, labels = inset_ax_roc.get_legend_handles_labels()
    fig.legend(handles, ['Safety-critical','GSSM', 'ACT', 'TTC2D'], 
            loc='lower center', ncol=4, bbox_to_anchor=(0.67, -0.13), frameon=False)
    
    return fig


def read_meta(path_processed, path_result):
    meta_all = pd.read_csv(path_processed + 'SHRP2/metadata_birdseye.csv').set_index('event_id')
    meta_all.loc[meta_all['event_category']=='Crash', 'severity_first'] = 3
    meta_all.loc[meta_all['event_category']=='Crash', 'severity_second'] = 0
    meta_all.loc[meta_all['event_category']=='NearCrash', 'severity_first'] = 2
    meta_all.loc[meta_all['event_category']=='NearCrash', 'severity_second'] = 0
    secondary_events = ['SecondaryCrash', 'SecondaryNearCrash']
    meta_all.loc[meta_all['event_category'].isin(secondary_events), 'severity_first'] = 0
    meta_all.loc[meta_all['event_category']=='SecondaryCrash', 'severity_second'] = 3
    meta_all.loc[meta_all['event_category']=='SecondaryNearCrash', 'severity_second'] = 2
    connected_events = ['Crash-Crash', 'Crash-NearCrash', 'NearCrash-Crash', 'NearCrash-NearCrash',
                        'NearCrash-CrashRelevant', 'CrashRelevant-NearCrash', 
                        'NearCrash-OtherConflict', 'Crash-OtherConflict',
                        'CrashRelevant-Crash', 'Crash-CrashRelevant']
    condition = meta_all['event_category'].isin(connected_events)
    first = meta_all['event_category'].apply(lambda x: x.split('-')[0] if '-' in x else x)
    second = meta_all['event_category'].apply(lambda x: x.split('-')[1] if '-' in x else x)
    meta_all.loc[condition&(first=='Crash'), 'severity_first'] = 3
    meta_all.loc[condition&(first=='NearCrash'), 'severity_first'] = 2
    meta_all.loc[condition&(meta_all['severity_first'].isna()), 'severity_first'] = 1
    meta_all.loc[condition&(second=='Crash'), 'severity_second'] = 3
    meta_all.loc[condition&(second=='NearCrash'), 'severity_second'] = 2
    meta_all.loc[condition&(meta_all['severity_second'].isna()), 'severity_second'] = 1

    meta_all.loc[meta_all['event_category']=='SafeBaseline', 'conflict'] = 'none'
    meta_all.loc[(meta_all['severity_first']<0.5)&(meta_all['severity_second']<0.5), 'conflict'] = 'none'
    condition = meta_all['severity_first']>=meta_all['severity_second']
    meta_all.loc[condition, 'conflict'] = meta_all.loc[condition, 'first']
    condition = meta_all['severity_second']>meta_all['severity_first']
    meta_all.loc[condition, 'conflict'] = meta_all.loc[condition, 'second']
    meta_all['conflict'] = [conflict_type[c] for c in meta_all['conflict'].values]

    meta_reconstructed = meta_all[(meta_all['ego_reconstructed'].astype(bool))&
                                  (meta_all['surrounding_reconstructed'].astype(bool))]

    event_categories = sorted(os.listdir(path_result + 'EventData/'))
    meta_events = pd.concat([pd.read_csv(path_result + 'EventData/' + f'{event_cat}/event_meta.csv') for event_cat in event_categories])
    meta_events = meta_events[meta_events['duration_enough']&(meta_events['conflict']!='none')].set_index('event_id')
    meta_events[['severity_first','severity_second','conflict']] = meta_reconstructed.loc[meta_events.index, ['severity_first','severity_second','conflict']].values

    environment = pd.concat([pd.read_csv(path_result + 'EventData/' + f'{event_cat}/environment.csv') for event_cat in event_categories])
    environment = environment.set_index('event_id').loc[meta_events.index]

    return meta_all, meta_reconstructed, meta_events, environment


def get_rank(eg_columns, conflict_warning, attribution, optimal_threshold, type='both'):
    warning_attribution = []
    non_warning_attribution = []
    for event_id in tqdm(conflict_warning.index.values):
        start_time, end_time = conflict_warning.loc[event_id][['danger_start','danger_end']].values/1000
        if conflict_warning.loc[event_id]['true_warning']>0.5:
            wattr = attribution[(attribution['event_id']==event_id)&
                                (attribution['time']>=start_time)&(attribution['time']<=end_time)]
            warning_attribution.append(wattr)
        if conflict_warning.loc[event_id]['num_true_non_warning']>0.5:
            nattr = attribution[(attribution['event_id']==event_id)&
                                (attribution['time']<start_time-3)]
            non_warning_attribution.append(nattr)
    warning_attribution = pd.concat(warning_attribution).reset_index(drop=True)
    non_warning_attribution = pd.concat(non_warning_attribution).reset_index(drop=True)

    if type=='non_warning' or type=='both':
        non_warning_attribution = non_warning_attribution[non_warning_attribution['intensity']<=optimal_threshold['threshold']]
        non_warning_statistics = pd.DataFrame(np.zeros((1,len(eg_columns))), columns=eg_columns)
        for idx in tqdm(range(len(non_warning_attribution)), desc='Non-warning attribution'):
            attrs = non_warning_attribution.iloc[idx][eg_columns]
            if np.all(attrs>=0):
                continue
            top3 = attrs[attrs<0].nsmallest(3)
            non_warning_statistics.loc[0,top3.index.values] = non_warning_statistics.loc[0,top3.index.values] + 1 #top3.values
        non_warning_statistics = non_warning_statistics.loc[0]
    else:
        non_warning_statistics = None

    if type=='warning' or type=='both':
        warning_attribution = warning_attribution[warning_attribution['intensity']>optimal_threshold['threshold']]
        warning_statistics = pd.DataFrame(np.zeros((1,len(eg_columns))), columns=eg_columns)
        for idx in tqdm(range(len(warning_attribution)), desc='Warning attribution'):
            attrs = warning_attribution.iloc[idx][eg_columns]
            if np.all(attrs<=0):
                continue
            top3 = attrs[attrs>0].nlargest(3)
            warning_statistics.loc[0,top3.index.values] = warning_statistics.loc[0,top3.index.values] + 1 #top3.values
        warning_statistics = warning_statistics.loc[0]
    else:
        warning_statistics = None
    
    return warning_statistics, non_warning_statistics


def settle_ax(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in')
    ax.set_yticks([ax.get_ylim()[0], ax.get_ylim()[1]])
    ax.set_yticklabels([])
    xmax = ax.get_xlim()[1]
    xticks = ax.get_xticks()
    if xticks[-1]>xmax:
        xticks = list(xticks[:-1])
        xticklabels = ax.get_xticklabels()[:-1]
        if xmax-xticks[-1]<xmax*0.02:
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticks(xticks+[xmax])
            ax.set_xticklabels(xticklabels+[''])

def get_sorted(statistics, proportion=0.9, number=None):
    statistics = statistics/statistics.sum()
    statistics = statistics.sort_values(ascending=False)
    if number is None:
        statistics = statistics[statistics.cumsum()<proportion]
    else:
        statistics = statistics[:number]
    return statistics

def plot_bars(statistics, ax):
    stat2plot = get_sorted(statistics, proportion=0.9, number=6)
    ax.barh(np.arange(len(stat2plot)), stat2plot.values[::-1],
            color=cmap(np.linspace(0.65, 0.15, len(stat2plot))), alpha=0.75, lw=0.35)
    xmax = ax.get_xlim()[1]
    for pos, label in zip(np.arange(len(stat2plot)), stat2plot.index[::-1]):
        if label=='eg_Sur lat speed':
            label = "eg_Surrounding object's lateral speed"
        if label=='eg_Sur lon speed':
            label = "eg_Surrounding object's longitudinal speed"
        if label=='eg_2D spacing direction':
            label = 'eg_Spacing direction'
        ax.text(xmax*0.99, pos, label.split('_')[1], ha='right', va='center', color='k')
    settle_ax(ax)


def draw_attribution(conflict_warning, attribution, meta_events, environment, optimal_threshold, eg_columns):
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.), constrained_layout=True, gridspec_kw={'wspace':0.1})
    ax_conflit = axes[0]
    ax_conflit.set_title('(a) Top factors in leading to and avoiding lateral conflicts', pad=15)
    remove_box(ax_conflit)
    ax_adj_danger = ax_conflit.inset_axes([0, 0.55, 0.45, 0.45])
    ax_adj_danger.set_title('Danger in adjacent lane', pad=3)
    filtered_warning = conflict_warning.loc[meta_events[(meta_events['conflict']=='Adjacent lane')].index.values]
    warning_statistics, non_warning_statistics = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='both')
    plot_bars(warning_statistics, ax_adj_danger)

    ax_adj_safe = ax_conflit.inset_axes([0.55, 0.55, 0.45, 0.45])
    ax_adj_safe.set_title('Safe in adjacent lane', pad=3)
    plot_bars(non_warning_statistics, ax_adj_safe)

    ax_cat_danger = ax_conflit.inset_axes([0, -0.05, 0.45, 0.45])
    ax_cat_danger.set_title('Danger during crossing/turning', pad=3)
    filtered_warning = conflict_warning.loc[meta_events[(meta_events['conflict']=='Crossing/turning')].index.values]
    warning_statistics, non_warning_statistics = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='both')
    plot_bars(warning_statistics, ax_cat_danger)

    ax_cat_safe = ax_conflit.inset_axes([0.55, -0.05, 0.45, 0.45])
    ax_cat_safe.set_title('Safe during crossing/turning', pad=3)
    plot_bars(non_warning_statistics, ax_cat_safe)

    ax_environment = axes[1]
    ax_environment.set_title('(b) Top factors in conflicts in adverse environments', pad=15)
    remove_box(ax_environment)
    ax_wea = ax_environment.inset_axes([0, 0.55, 0.45, 0.45])
    ax_wea.set_title('Raining weather', pad=3)
    filtered_warning = conflict_warning.loc[environment[(environment['weather']=='Raining')|
                                                        (environment['weather']=='Mist/Light Rain')].index.values]
    warning_statistics, _ = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='warning')
    plot_bars(warning_statistics, ax_wea)

    ax_road = ax_environment.inset_axes([0.55, 0.55, 0.45, 0.45])
    ax_road.set_title('Not dry road', pad=3)
    filtered_warning = conflict_warning.loc[environment[(environment['surfaceCondition']!='Dry')].index.values]
    warning_statistics, _ = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='warning')
    plot_bars(warning_statistics, ax_road)

    ax_light = ax_environment.inset_axes([0, -0.05, 0.45, 0.45])
    ax_light.set_title('Not in daylight', pad=3)
    filtered_warning = conflict_warning.loc[environment[(environment['lighting']!='Daylight')].index.values]
    warning_statistics, _ = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='warning')
    plot_bars(warning_statistics, ax_light)

    ax_traffic = ax_environment.inset_axes([0.55, -0.05, 0.45, 0.45])
    ax_traffic.set_title('LOS D Unstable flow', pad=3)
    filtered_warning = conflict_warning.loc[environment[(environment['trafficDensity']=='Level-of-service D: Unstable flow - temporary restrictions substantially slow driver')].index.values]
    warning_statistics, _ = get_rank(eg_columns, filtered_warning, attribution, optimal_threshold, type='warning')
    plot_bars(warning_statistics, ax_traffic)

    for ax in [ax_cat_danger, ax_cat_safe, ax_light, ax_traffic]:
        ax.set_xlabel('Frequency of being top 3 factors')

    return fig, axes